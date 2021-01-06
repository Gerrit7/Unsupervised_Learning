import torch
import torch.nn as nn
import os
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

import AverageMeter

def train(model, optimizer, criterion, train_loader, device, task):
    ''' training function
    :param model: the model to train
    :param optimizer: optimizer used in training
    :param criterion: loss function
    :param train_loader: DataLoader of training set
    :param device: cpu or cuda
    :param task: task of current dataset, binary-class/multi-class/multi-label, binary-class

    '''

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
    
    return optimizer, loss

def loss_soft_reg_ep(preds, labels, soft_labels, device, num_classes):
    prob = F.softmax(preds, dim=1)
    prob_avg = torch.mean(prob, dim=0)
    p = torch.ones(num_classes).to(device) / num_classes

    L_c = -torch.mean(torch.sum(soft_labels * F.log_softmax(preds, dim=1), dim=1))   # Soft labels
    L_p = -torch.sum(torch.log(prob_avg) * p)
    L_e = -torch.mean(torch.sum(prob * F.log_softmax(preds, dim=1), dim=1))

    loss = L_c + 0.8 * L_p + 0.4* L_e
    return prob, loss

def train_CrossEntropy(model, optimizer, train_loader, epoch, num_classes, device, unlabeled_indexes):
    train_loss = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    counter = 1
    results = np.zeros((len(train_loader.dataset), num_classes), dtype=np.float32)
    for imgs, img_pslab, labels, soft_labels, index in train_loader:
        images, images_pslab, labels, soft_labels = imgs.to(device), img_pslab.to(device), labels.to(device), soft_labels.to(device)

        # compute output
        outputs = model(images)

        prob, loss = loss_soft_reg_ep(outputs, labels, soft_labels, device, num_classes)

        results[index.detach().numpy().tolist()] = prob.cpu().detach().numpy().tolist()

        prec1, prec5 = accuracy_v2(outputs, labels, top=[1, 1])
        train_loss.update(loss.item(), images.size(0))
        top1.update(prec1.item(), images.size(0))
        top5.update(prec5.item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if counter % 15 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {:.0f}%, Learning rate: {:.6f}'.format(
                epoch, counter * len(images), len(train_loader.dataset),
                        100. * counter / len(train_loader), loss.item(),
                        prec1, optimizer.param_groups[0]['lr']))
        counter = counter + 1

        # update soft labels
        train_loader.dataset.update_labels(results, unlabeled_indexes)

    return train_loss.avg, top5.avg, top1.avg

def val(model, val_loader, device, val_auc_list, task, dir_path, epoch, auc_old, epoch_old, optimizer, loss):
    ''' validation function
    :param model: the model to validate
    :param val_loader: DataLoader of validation set
    :param device: cpu or cuda
    :param val_auc_list: the list to save AUC score of each epoch
    :param task: task of current dataset, binary-class/multi-class/multi-label, binary-class
    :param dir_path: where to save model
    :param epoch: current epoch

    '''

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

            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, outputs), 0)

        y_true = y_true.cpu().numpy()
        y_score = y_score.detach().cpu().numpy()
        auc = getAUC(y_true, y_score, task)
        val_auc_list.append(auc)

    state = {
        'net': model.state_dict(),
        'auc': auc,
        'epoch': epoch,
        'optimizer': optimizer,
        'loss': loss
    }
    
    if auc > auc_old:
        print("old auc: ", auc_old)
        print("new auc: ", auc)
        
        path = os.path.join(dir_path, 'ckpt_%d_auc_%.5f.pth' % (epoch, auc))
        torch.save(state, path)
        if epoch>1:
            del_path = os.path.join(dir_path, 'ckpt_%d_auc_%.5f.pth' % (epoch_old, auc_old))
            os.remove(del_path) 

    return (epoch, auc)

def test(model, split, data_loader, device, flag, task, output_root=None):
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
            output_dir = os.path.join(output_root, flag)
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            output_path = os.path.join(output_dir, '%s.csv' % (split))
            save_results(y_true, y_score, output_path)


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

def accuracy_v2(preds, labels, top=[1,5]):
    """Compute the precision@k for the specified values of k"""
    result = []
    maxk = max(top)
    batch_size = preds.size(0)

    _, pred = preds.topk(maxk, 1, True, True)
    pred = pred.t() # pred[k-1] stores the k-th predicted label for all samples in the batch.
    correct = pred.eq(labels.view(1,-1).expand_as(pred))

    for k in top:
        correct_k = correct[:k].view(-1).float().sum(0)
        result.append(correct_k.mul_(100.0 / batch_size))

    return result

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



def load_checkpoint(model, optimizer, val_auc_list, filename='checkpoint.pth.tar'):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        print("loading epoch")
        start_epoch = checkpoint['epoch']+1
        print("loading model")
        model.load_state_dict(checkpoint['net'])
        print("loading optimizer")
        optimizer = checkpoint['optimizer']
        print("loading loss")
        loss = checkpoint['loss']
        print("loading auc_list")
        val_auc_list = checkpoint['auc']
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, loss, start_epoch, val_auc_list


def createPseudoLabel(model, data_loader, device, task, output_root=None):
    ''' pseudo labeling function
    :param model: the model to test
    :param data_loader: DataLoader of data
    :param device: cpu or cuda
    :param task: task of current dataset, binary-class/multi-class/multi-label, binary-class

    '''
    model.eval()
    y_true = torch.tensor([]).to(device)
    y_score = torch.tensor([]).to(device)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            for single_input, single_target in zip(inputs, targets):
                output = model(torch.unsqueeze(single_input.to(device),0))

                if task == 'multi-label, binary-class':
                    m = nn.Sigmoid()
                    output = m(output).to(device)
                else:
                    m = nn.Softmax(dim=1)
                    output = m(output).to(device)
                    
                
                #pseudolabeled_data = (single_input, output.index(max(output)), single_target)
                pseudolabeled_data = (output.max(), output.argmax(), single_target)
                
                #if output.max() > 0.7:

        
                
        #auc = getAUC(y_true, y_score, task)
        #acc = getACC(y_true, y_score, task)
        #print('AUC: %.5f ACC: %.5f' % (auc, acc))
