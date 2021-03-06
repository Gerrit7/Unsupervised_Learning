import glob
import os
import shutil
from shutil import copyfile

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.nn import functional as F
from tqdm import trange

import cv2
import matplotlib.pyplot as plt

from classes.PrepareData import createDataLoader
from classes.CreateModel import createNewCNN, load_checkpoint, createLossFunction, createOptimizer, createScheduler

def defineOperators(n_channels, n_classes, dataloader, dir_path, inputs, device):

    model, image_size = createNewCNN(inputs['net_input'], n_channels, n_classes, device)
    criterion = createLossFunction(inputs['loss_function'])
    optimizer = createOptimizer(inputs['optimizer'], model, inputs['momentum'], inputs['weight_decay'], inputs['lr'])
    scheduler = createScheduler(optimizer, len(dataloader), inputs['milestone_count'], inputs['decayLr'])

    # check wheather a training session has already startet for this dataset
    if os.path.isdir(dir_path) and len(os.listdir(dir_path)) != 0:
        list_of_files = glob.glob(dir_path + "/ckpt*.pth")
        latest_file = max(list_of_files, key=os.path.getctime)
        filename = latest_file
        model, optimizer, scheduler, loss, start_epoch, val_auc_list = load_checkpoint(model, optimizer, scheduler, filename)
        epoch_old = start_epoch
        auc_old = val_auc_list[-1]
    else:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        start_epoch = 0
        epoch_old = 0
        auc_old = 0
        val_auc_list = []

    # push optimizer and model to device (cpu/gpu)
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    model.to(device)

    return start_epoch, model, criterion, optimizer, scheduler, epoch_old, auc_old, val_auc_list

def train_labeled(model, optimizer, criterion, train_loader, task, device):
    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs.to(device))
        
        if task == 'multi-label, binary-class':
            targets = targets.to(torch.float32).to(device)
            loss = criterion(outputs, targets)
        else:
            targets = targets.squeeze().long().to(device)
            loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()
    
    return model, optimizer, criterion, loss


def val_labeled(model, optimizer, scheduler, val_loader, task, val_auc_list, dir_path, epoch, auc_old, epoch_old, loss, device):

    model.eval()
    y_true = torch.tensor([]).to(device)
    y_score = torch.tensor([]).to(device)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            outputs = model(inputs.to(device))

            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32).to(device)
                m = nn.Sigmoid()
                outputs = m(outputs).to(device)
            else:
                targets = targets.squeeze().long().to(device)
                m = nn.Softmax(dim=1)
                outputs = m(outputs).to(device)
                targets = targets.float().resize_(len(targets), 1)
            #print("outputs: ", outputs)
            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, outputs), 0)

        y_true = y_true.cpu().numpy()
        y_score = y_score.detach().cpu().numpy()
        auc = getAUC(y_true, y_score, task)
        val_auc_list.append(auc)
    
    # if auc <= auc_old:
    #     if os.path.isdir(dir_path) and len(os.listdir(dir_path)) != 0:
    #         list_of_files = glob.glob(dir_path + "/*")
    #         latest_file = max(list_of_files, key=os.path.getctime)
    #         filename = latest_file
    #         checkpoint = torch.load(filename)
    #         model.load_state_dict(checkpoint['net'])
    #         optimizer.load_state_dict(checkpoint['optimizer'])
    
    
    if auc > auc_old:
        state = {
            'net': model.state_dict(),
            'auc_list': val_auc_list,
            'epoch': epoch,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'loss': loss
        }

        path = os.path.join(dir_path, 'ckpt_%d_auc_%.5f.pth' % (epoch, auc))
        torch.save(state, path)
        if epoch>0:
            os.remove(os.path.join(dir_path, 'ckpt_%d_auc_%.5f.pth' % (epoch_old, auc_old))) 
    else: 
        
        auc = auc_old
        epoch = epoch_old
    

    return epoch, auc


def test_labeled(split, model, data_loader, flag, split_size, task, device, output_root=None):
    ''' testing function
    :param model: the model to test
    :param split: the data to test, 'train/val/test'
    :param data_loader: DataLoader of data
    :param device: cpu or cuda
    :param flag: subset name
    :param task: task of current dataset, binary-class/multi-class/multi-label, binary-class

    '''

    model.eval()
    y_true = torch.tensor([]).to(device)
    y_score = torch.tensor([]).to(device)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            outputs = model(inputs.to(device))

            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32).to(device)
                m = nn.Sigmoid()
                outputs = m(outputs).to(device)
            else:
                targets = targets.squeeze().long().to(device)
                m = nn.Softmax(dim=1)
                outputs = m(outputs).to(device)
                targets = targets.float().resize_(len(targets), 1)

            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, outputs), 0)

        y_true = y_true.cpu().numpy()
        y_score = y_score.detach().cpu().numpy()
        auc = getAUC(y_true, y_score, task)
        acc = getACC(y_true, y_score, task)
        print('%s AUC: %.5f ACC: %.5f' % (split, auc, acc))

        if output_root is not None:
            output_dir = os.path.join(output_root, flag + "_" + '%.2f' % split_size)
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            output_path = os.path.join(output_dir, '%s.csv' % (split))
            save_results(y_true, y_score, output_path)

    return auc, acc


def getAUC(y_true, y_score, task):
    '''AUC metric.
    :param y_true: the ground truth labels, shape: (n_samples, n_classes) for multi-label, and (n_samples,) for other tasks
    :param y_score: the predicted score of each class, shape: (n_samples, n_classes)
    :param task: the task of current dataset

    '''
    if task == 'binary-class':
        y_score = y_score[:,-1]
        return roc_auc_score(y_true, y_score)
    elif task == 'multi-label, binary-class':
        auc = 0
        for i in range(y_score.shape[1]):
            label_auc = roc_auc_score(y_true[:, i], y_score[:, i])
            auc += label_auc
        return auc / y_score.shape[1]
    else:
        auc = 0
        zero = np.zeros_like(y_true)
        one = np.ones_like(y_true)
        for i in range(y_score.shape[1]):
            y_true_binary = np.where(y_true == i, one, zero)
            y_score_binary = y_score[:, i]
            auc += roc_auc_score(y_true_binary, y_score_binary)
        return auc / y_score.shape[1]


def getACC(y_true, y_score, task, threshold=0.5):
    '''Accuracy metric.
    :param y_true: the ground truth labels, shape: (n_samples, n_classes) for multi-label, and (n_samples,) for other tasks
    :param y_score: the predicted score of each class, shape: (n_samples, n_classes)
    :param task: the task of current dataset
    :param threshold: the threshold for multilabel and binary-class tasks

    '''
    if task == 'multi-label, binary-class':
        zero = np.zeros_like(y_score)
        one = np.ones_like(y_score)
        y_pre = np.where(y_score < threshold, zero, one)
        acc = 0
        for label in range(y_true.shape[1]):
            label_acc = accuracy_score(y_true[:, label], y_pre[:, label])
            acc += label_acc
        return acc / y_true.shape[1]
    elif task == 'binary-class':
        y_pre = np.zeros_like(y_true)
        for i in range(y_score.shape[0]):
            y_pre[i] = (y_score[i][-1] > threshold)
        return accuracy_score(y_true, y_pre)
    else:
        y_pre = np.zeros_like(y_true)
        for i in range(y_score.shape[0]):
            y_pre[i] = np.argmax(y_score[i])
        return accuracy_score(y_true, y_pre)


def save_results(y_true, y_score, outputpath):
    '''Save ground truth and scores
    :param y_true: the ground truth labels, shape: (n_samples, n_classes) for multi-label, and (n_samples,) for other tasks
    :param y_score: the predicted score of each class, shape: (n_samples, n_classes)
    :param outputpath: path to save the result csv

    '''
    idx = []

    idx.append('id')

    for i in range(y_true.shape[1]):
        idx.append('true_%s' % (i))
    for i in range(y_score.shape[1]):
        idx.append('score_%s' % (i))

    df = pd.DataFrame(columns=idx)
    for id in range(y_score.shape[0]):
        dic = {}
        dic['id'] = id
        for i in range(y_true.shape[1]):
            dic['true_%s' % (i)] = y_true[id][i]
        for i in range(y_score.shape[1]):
            dic['score_%s' % (i)] = y_score[id][i]

        df_insert = pd.DataFrame(dic, index = [0])
        df = df.append(df_insert, ignore_index=True)

    df.to_csv(outputpath, sep=',', index=False, header=True, encoding="utf_8_sig")


def evalModel(dir_path, model, train_loader, val_loader, test_loader, task, auc_list, index, inputs, device):
    #*************************** Evaluate trained Model **************************************
    print('==> Testing model...')
    restore_model_path = os.path.join(
        dir_path, 'ckpt_%d_auc_%.5f.pth' % (index, auc_list[index]))
    #restore_model_path = os.path.join(dir_path,)
    
    model.load_state_dict(torch.load(restore_model_path)['net'])
    auc_train, acc_train = test_labeled('train', model, train_loader, inputs['dataset'], inputs['train_size'], task, device, output_root=inputs['output_root'])
    auc_val, acc_val = test_labeled('val', model, val_loader, inputs['dataset'],  inputs['train_size'], task, device, output_root=inputs['output_root'])
    auc_test, acc_test = test_labeled('test', model, test_loader, inputs['dataset'],  inputs['train_size'], task, device, output_root=inputs['output_root'])
        
    # save_best_model_path = os.path.join( 
    #     os.path.join(inputs['output_root'], inputs['dataset'] + "_" + str('%.2f' % ['train_size'])), 'ckpt_%d_auc_%.5f.pth' % (index, auc_list[index]))
    # copyfile(restore_model_path, save_best_model_path)
    #shutil.rmtree(dir_path)

    return auc_train, acc_train, auc_val, acc_val, auc_test, acc_test

# Pseudo Labeling Data
def step(data, model, device, showImage=False):
    
    m = nn.Softmax(dim=1)
    outputs = m(model(data.to(device)))   

    if showImage == True:
        for i in range(len(data)):
            plt.figure()
            plt.imshow(data[i][0])
            plt.show()
            input("Press enter to continue!")
            plt.close()
    return outputs

def hardlabels(dataloader, model, device, info, image_size = 28, pos_thresh=0.7, neg_thresh = 0.05):
    # for i in range(len(dataloader.dataset)):
    #     print(dataloader.dataset[i][1])
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])
    num_elements = len(dataloader.dataset)
    num_batches = len(dataloader)
    batch_size = dataloader.batch_size
    print("length of data in pseudo-labeling dataloader: ", num_elements)

    if task == 'multi-label, binary-class' or  task == 'binary_class':
        predictions = torch.zeros(num_elements, n_classes).to(device)
        images = torch.zeros(num_elements, n_channels, image_size, image_size)
        labels = torch.zeros(num_elements, n_classes)
    else:
        predictions = torch.zeros(num_elements, 1).to(device)
        images = torch.zeros(num_elements, n_channels, image_size, image_size)
        labels = torch.zeros(num_elements, 1)
    # predictions = torch.tensor([]).to(device)
    # images = torch.tensor([])
    # labels = torch.tensor([])

    

    predictions_thresh = torch.tensor([]).to(device)
    images_thresh = torch.tensor([])
    labels_thresh = torch.tensor([])

    auc = 0
    for i, batch in enumerate(dataloader):
        start = i*batch_size
        end = start + batch_size
        if i == num_batches - 1:
            end = num_elements
        data, targets = batch
        outputs = step(data, model, device)
        
        if task == 'multi-label, binary-class' or 'binary_class':
            pos_thresh = 0.5
            taken_preds = ((outputs>pos_thresh) + (outputs<neg_thresh))*1
            vals , hardlabels = torch.max(outputs.data , 1)
            hardlabels = hardlabels.reshape(end-start, 1)
        else:
            taken_preds = ((outputs>pos_thresh) + (outputs<neg_thresh))*1
            hardlabels = ((outputs>pos_thresh))*1

        indices = [idx for idx,value in enumerate(taken_preds.tolist()) if np.asarray(taken_preds.to(torch.device("cpu")))[idx].all()]
        bad_indices = [idx for idx,value in enumerate(taken_preds.tolist()) if not np.asarray(taken_preds.to(torch.device("cpu")))[idx].all()]
        
        predictions_thresh = torch.cat((predictions_thresh, hardlabels[indices]), 0)
        images_thresh = torch.cat((images_thresh, data[indices]), 0)
        labels_thresh = torch.cat((labels_thresh, targets[indices]), 0)

        # print("Outputs: ", outputs.size(), outputs)
        # print("Hardlabels: ", hardlabels.size(),hardlabels)
        # print("Targets: ", targets.size(),targets)
        # print("Indices: ", len(indices),indices)
        # input("press enter!")
        predictions[start:end] = hardlabels
        images[start:end] = data
        labels[start:end] = targets

        # print("correct labeled targets without threshold: ", accuracy_score(targets.to(torch.device("cpu")), hardlabels.to(torch.device("cpu")))*100, "%")
        # print("correct labeled targets with threshold: ", get)
        
        # print(targets.size())
        # print(hardlabels.size())
        # print(targets[indices].size())
        # print(hardlabels[indices].size())
        # input("press enter!")

        print("correct labeled targets without threshold: ", accuracy_score(targets.to(torch.device("cpu")), hardlabels.to(torch.device("cpu")))*100, "%")
        print("correct labeled targets with threshold: ", accuracy_score(targets[indices].to(torch.device("cpu")), hardlabels[indices].to(torch.device("cpu")))*100, "%")



    # print("Images: ", images.size())
    # print("Labels: ", labels.size())
    # print("Preds: ", predictions.size())

    # print("Images Thresh: ", images_thresh.size())
    # print("Labels Thresh: ", labels_thresh.size())
    # print("Preds Thresh: ", predictions_thresh.size())

    print("predictions greater than threshold :", len(predictions_thresh), "/" ,len(predictions))
    # print("correct labeled targets without threshold: ", getAUC(labels.to(torch.device("cpu")), predictions.to(torch.device("cpu")), task)*100, "%")
    # print("correct labeled targets with threshold: ", getAUC(labels_thresh.to(torch.device("cpu")), predictions_thresh.to(torch.device("cpu")), task)*100, "%")
    print("correct labeled targets without threshold: ", accuracy_score(labels.to(torch.device("cpu")), predictions.to(torch.device("cpu")))*100, "%")
    print("correct labeled targets with threshold: ", accuracy_score(labels_thresh.to(torch.device("cpu")), predictions_thresh.to(torch.device("cpu")))*100, "%")

    return images_thresh, predictions, predictions_thresh.to(torch.device("cpu")), labels, labels_thresh
