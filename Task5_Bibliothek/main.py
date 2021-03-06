import glob
import os
import shutil
from shutil import copyfile

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import math
import numpy as np
from pathlib import Path
import sys
import csv
import torch
from torch.nn import functional as F
import torch.nn as nn
from tqdm import trange
import torch.utils.data as data

from models.models import EfficientNet, ResNet18, ResNet50
from classes.CreateModel import createNewCNN, load_checkpoint, createLossFunction, createOptimizer, createScheduler
from classes.Operations import train_labeled, val_labeled, test_labeled, evalModel, defineOperators, hardlabels
from classes.PrepareData import PrepareData, splitDataset, createDataLoader, ConcatDataset
from medmnist.info import INFO
from functions.dataset_pseudo import PathMNIST, ChestMNIST, DermaMNIST, OCTMNIST, PneumoniaMNIST, RetinaMNIST, \
    BreastMNIST, OrganMNISTAxial, OrganMNISTCoronal, OrganMNISTSagittal


def main(dataset_name,
         data_root,
         output_root,
         num_epoch,
         batch_size,
         learning_rate,
         momentum,
         train_size,
         weight_decay,
         net_input,
         count_students,
         mode,
         task_input,
         optimizer_input,
         decayLr,
         milestone_count,
         loss_function,
         augmentations,
         train_teacher_dependent,
         download):

    inputs = {
        'dataset': dataset_name,
        'data_root': data_root,
        'output_root':output_root,
        'num_epochs': num_epoch,
        'batch_size': batch_size,
        'lr': learning_rate,
        'momentum': momentum,
        'train_size': train_size,
        'weight_decay': weight_decay,
        'net_input': net_input,
        'count_students': count_students,
        'mode': mode,
        'task': task_input,
        'optimizer': optimizer_input,
        'decayLr': decayLr,
        'milestone_count': milestone_count,
        'loss_function': loss_function,
        'augs': augmentations,
        'train_teacher_dependent': train_teacher_dependent,
        'download': download
    }
    download = False
    # ************************************** predefine some values *****************************************
    start_epoch = 0
    if dataset_name != 'cifar10':
        
        info = INFO[dataset_name]
        task = info['task']
        n_channels = info['n_channels']
        n_classes = len(info['label'])
        n_train = info['n_samples']['train']
        n_val = info['n_samples']['val']
        n_test = info['n_samples']['test']
        val_auc_list = []
        epoch_old = 0
        auc_old = 0
        loss = 0
        flag = dataset_name
        dataset_name = dataset_name + "_" + '%.2f' % train_size
        dir_path = os.path.join(output_root, '%s' % (dataset_name))
    
    elif dataset_name == 'cifar10':
        print('cifar10')
    else:
        print('nothing')


    # ************************************** train on gpu if possible **************************************
    
    if torch.cuda.is_available():     
        print('used GPU: ' + torch.cuda.get_device_name(0))
        device = torch.device("cuda:0")
        kwar = {'num_workers': 8, 'pin_memory': True}
        cpu = torch.device("cpu")
    
    else:
        print("Warning: CUDA not found, CPU only.")
        device = torch.device("cpu")
        kwar = {}
        cpu = torch.device("cpu")
    

    # ************************************** setting up mode ***********************************************
    
    if mode.get() == False: 
        mode = "training"
    else:
        mode = "prediction"

    
    # ************************************** prepare data for training *************************************
    prepareClass = PrepareData(flag, data_root, output_root, net_input, download=True)
    train_transform = prepareClass.createTransform(image_size=32, augmentations=augmentations)
    val_transform = prepareClass.createTransform(image_size=32, augmentations=augmentations)
    test_transform = prepareClass.createTransform(image_size=32, augmentations=augmentations)
    
    #train_dataset, train_subset_labeled, train_dataset_labeled, train_subset_unlabeled, train_dataset_unlabeled = prepareClass.prepareDataSet('train', train_transform, train_size)
    train_dataset = prepareClass.prepareDataSet('train', train_transform, train_size)
    train_loader   = createDataLoader(train_dataset, batch_size)
    train_dataset_labeled, train_subset_labeled, train_dataset_unlabeled, train_subset_unlabeled = splitDataset(train_dataset, train_loader, n_classes, n_train, train_size, task)

    train_loader_labeled   = createDataLoader(train_subset_labeled, batch_size)
    if len(train_dataset_unlabeled)>0:
        train_loader_unlabeled = createDataLoader(train_subset_unlabeled, batch_size)
    
    #print(len(train_loader_labeled), len(train_loader_unlabeled))
    
    #val_dataset, val_subset_labeled, val_dataset_labeled, val_subset_unlabeled, val_dataset_unlabeled = prepareClass.prepareDataSet('val', val_transform, 1)
    val_dataset = prepareClass.prepareDataSet('val', val_transform, 1)
    val_loader   = createDataLoader(train_dataset, batch_size)
    val_dataset_labeled, val_subset_labeled, val_dataset_unlabeled, val_subset_unlabeled = splitDataset(val_dataset, val_loader, n_classes, n_val, train_size, task)

    val_loader_labeled   = createDataLoader(val_subset_labeled, batch_size)
    #if len(val_dataset_unlabeled)>0:
        #val_loader_unlabeled = createDataLoader(val_subset_unlabeled, batch_size)

    #test_dataset, test_subset_labeled, test_dataset_labeled, test_subset_unlabeled, test_dataset_unlabeled = prepareClass.prepareDataSet('test', test_transform, 1)
    test_dataset = prepareClass.prepareDataSet('test', test_transform, 1)
    test_loader   = createDataLoader(test_dataset, batch_size)
    test_dataset_labeled, test_subset_labeled, test_dataset_unlabeled, test_subset_unlabeled = splitDataset(test_dataset, test_loader, n_classes, n_test, train_size, task)

    
    test_loader_labeled   = createDataLoader(test_subset_labeled, batch_size)
    #if len(test_dataset_unlabeled)>0:
        #test_loader_unlabeled = createDataLoader(test_subset_unlabeled, batch_size)
    
    print('Train: ', len(train_subset_labeled), ', Valid: ', len(val_subset_labeled), ', Test: ', len(test_subset_labeled))



# ************************************** Training  ***************************************************************************************************
    if task_input == "BaseLine":
        print("==> Baseline-Training...")

        # creating model
        start_epoch, model, criterion, optimizer, scheduler, epoch_old, auc_old, val_auc_list = defineOperators(n_channels, n_classes, train_loader_labeled, dir_path, inputs, device)
        # training model
        for epoch in trange(start_epoch, num_epoch):
            model, optimizer, criterion, loss = train_labeled(model, optimizer, criterion, train_loader_labeled, task, device)
            epoch, auc = val_labeled(model, optimizer, scheduler, val_loader_labeled, task, val_auc_list, dir_path, epoch, auc_old, epoch_old, loss, device)
            auc_old = auc
            epoch_old = epoch
        auc_list = np.array(val_auc_list)
        index  = auc_list.argmax()
        print('epoch %s is the best model' % (index))
        
        savepath = os.path.join(dir_path, inputs['dataset'] + '%.2f' % train_size + "_bestmodel.pth")
        torch.save(model.state_dict(), savepath)
        # evaluate model
        auc_train, acc_train, auc_val, acc_val, auc_test, acc_test = evalModel(dir_path, model, train_loader_labeled, val_loader_labeled, test_loader_labeled, task, auc_list, index, inputs, device)

        labeling_accuracy_complete = '-'
        labeling_accuracy_thresh = '-'
        student_number = 'base'
        thresh_preds = '-'
        comp_preds = '-'
    
    elif task_input == "NoisyStudent":
        print("==> NoisyStudent-Training...")

    elif task_input == "MTSS":
        print("==> MTSS-Training...")

# ************************************** PseudoLabeling  *************************************************************************************************
    elif task_input == "Pseudolabel":
        print("==> Pseudolabel-Training...")
 
        labeled_inputs = torch.tensor([])
        labeled_targets = torch.tensor([])
        
        for batch_idx, (train_inputs, train_targets) in enumerate(train_loader_labeled):
            labeled_inputs = torch.cat((labeled_inputs, train_inputs), 0)
            labeled_targets = torch.cat((labeled_targets, train_targets), 0)

        # ************************************** Create Models  *****************************************
        # create teacher model
        model, image_size = createNewCNN(inputs['net_input'], n_channels, n_classes, device)
        
        # load model parameters
        if os.path.isdir(dir_path) and len(os.listdir(dir_path)) != 0:
            list_of_files = glob.glob(dir_path + "/*bestmodel.pth")
            latest_file = max(list_of_files, key=os.path.getctime)
            model.load_state_dict(torch.load(Path(latest_file)))
            #model, optimizer, scheduler, loss, start_epoch, val_auc_list = load_checkpoint(model, optimizer, scheduler, latest_file)
        else:
            sys.exit("No teacher net is pretrained!")

        for student_number in range(count_students):
            val_auc_list = []
            epoch_old = 0
            auc_old = 0
            loss = 0
            start_epoch = 0
            
            # start pseudolabeling with teacher model
            thresh_inputs, comp_preds, thresh_preds, comp_targets, thresh_targets = hardlabels(train_loader_unlabeled, model, device, info, image_size= image_size, )
            
            print(type(thresh_inputs))
            print(thresh_inputs.size())
            print(type(thresh_targets))
            print(thresh_targets.size())

            print(type(labeled_inputs))
            print(labeled_inputs.size())
            print(type(labeled_targets))
            print(labeled_targets.size())

            combined_inputs = np.concatenate((np.asarray(labeled_inputs), np.asarray(thresh_inputs))) 
            combined_targets = np.concatenate((np.asarray(labeled_targets), np.asarray(thresh_targets)))
            combined_inputs = torch.from_numpy(combined_inputs)
            combined_targets = torch.from_numpy(combined_targets)
            #np.savez(os.path.join(dir_path, 'combined_' + inputs['dataset'] +'.npz'), inputs=combined_inputs, targets=combined_targets)
            
            pseudo_dataset = data.TensorDataset(combined_inputs, combined_targets)
            combined_dataloader = data.DataLoader(dataset=pseudo_dataset, batch_size=batch_size, shuffle=True)

            # create student model
            model, image_size = createNewCNN(inputs['net_input'], n_channels, n_classes, device)
            criterion = createLossFunction(inputs['loss_function'])
            optimizer = createOptimizer(inputs['optimizer'], model, inputs['momentum'], inputs['weight_decay'], inputs['lr'])
            scheduler = createScheduler(optimizer, len(combined_dataloader), inputs['milestone_count'], inputs['decayLr'])

            #training model
            student_path = os.path.join(dir_path, "training_"+ str(student_number))
            if os.path.isdir(student_path) == True:
                shutil.rmtree(student_path)  
            os.mkdir(student_path)
            num_elements = len(combined_dataloader.dataset)
            print("length of data in training dataloader: ", num_elements)
            for epoch in trange(start_epoch, num_epoch):
                model, optimizer, criterion, loss = train_labeled(model, optimizer, criterion, combined_dataloader, task, device)
                epoch, auc = val_labeled(model, optimizer, scheduler, val_loader_labeled, task, val_auc_list, student_path, epoch, auc_old, epoch_old, loss, device)
                auc_old = auc
                epoch_old = epoch
            auc_list = np.array(val_auc_list)
            index = auc_list.argmax()
            print('epoch %s is the best model' % (index))
            
            # evaluate model
            auc_train, acc_train, auc_val, acc_val, auc_test, acc_test = evalModel(student_path, model, combined_dataloader, val_loader_labeled, test_loader_labeled, task, auc_list, index, inputs, device)

            labeling_accuracy_complete = accuracy_score(comp_targets.to(cpu), comp_preds.to(cpu))
            labeling_accuracy_thresh = accuracy_score(thresh_targets.to(cpu), thresh_preds.to(cpu))
    
    
    
    # save results in csv file
    file_exists = os.path.isfile(os.path.join(dir_path, 'results.csv'))
    with open(os.path.join(dir_path, 'results.csv'), 'a+', newline='') as csvfile:
        fieldnames = [  'training number', 
                        'labeled data', 
                        'count of labeled data',
                        'labeling accuracy complete',
                        'labeling accuracy threshold',
                        'AUC Train', 
                        'ACC Train',
                        'AUC Validation', 
                        'ACC Validation',
                        'AUC Test', 
                        'ACC Test']
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()  # file doesn't exist yet, write a header
        writer.writerow({   'training number': student_number, 
                            'labeled data': train_size,
                            'count of labeled data': [len(thresh_preds), "/" ,len(comp_preds)],
                            'labeling accuracy complete': labeling_accuracy_complete,
                            'labeling accuracy threshold' : labeling_accuracy_thresh,
                            'AUC Train': auc_train, 
                            'ACC Train': acc_train,
                            'AUC Validation': auc_val, 
                            'ACC Validation': auc_val,
                            'AUC Test': auc_test, 
                            'ACC Test': acc_test})


       