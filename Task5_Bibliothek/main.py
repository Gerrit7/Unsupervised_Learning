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
import torch
from torch.nn import functional as F
import torch.nn as nn
from tqdm import trange


from classes.CreateModel import CreateModel, LossFunction, Optimizer
from classes.Operations import train_labeled, val_labeled, test_labeled
from classes.PrepareData import PrepareData, splitDataset, createDataLoader
from classes.BaseLine import saveBestModel, defineOperators
from classes.PseudoLabel import PseudoLabel
from medmnist.info import INFO


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
        val_auc_list = []
        epoch_old = 0
        auc_old = 0
        loss = 0
        flag = dataset_name
        dataset_name = dataset_name + "_" + '%.2f' % train_size
        dir_path = os.path.join(output_root, '%s_checkpoints' % (dataset_name))
    
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
    
    train_dataset, train_dataset_labeled, train_indices_labeled, train_dataset_unlabeled, train_indices_unlabeled = prepareClass.prepareDataSet('train', train_transform, train_size)
    train_loader_labeled   = createDataLoader(train_dataset_labeled, batch_size)
    train_loader_unlabeled = createDataLoader(train_dataset_unlabeled, batch_size)
    #print(len(train_loader_labeled), len(train_loader_unlabeled))
    
    val_dataset, val_dataset_labeled, val_indices_labeled, val_dataset_unlabeled, val_indices_unlabeled = prepareClass.prepareDataSet('val', val_transform, 1)
    val_loader_labeled   = createDataLoader(val_dataset_labeled, batch_size)
    #val_loader_unlabeled = createDataLoader(val_dataset_unlabeled, batch_size)

    test_dataset, test_dataset_labeled, test_indices_labeled, test_dataset_unlabeled, test_indices_unlabeled = prepareClass.prepareDataSet('test', test_transform, 1)
    test_loader_labeled   = createDataLoader(test_dataset_labeled, batch_size)
    #test_loader_unlabeled = createDataLoader(test_dataset_unlabeled, batch_size)
    
    print('Train: ', len(train_dataset_labeled), ', Valid: ', len(val_dataset), ', Test: ', len(test_dataset))

    # ************************************** Training  *****************************************************
    if task_input == "BaseLine":
        print("==> Baseline-Training...")

        # creating model
        start_epoch, model, criterion, optimizer, scheduler, epoch_old, auc_old = defineOperators(n_channels, n_classes, train_loader_labeled, dir_path, inputs, device)
        # training model
        for epoch in trange(start_epoch, num_epoch):
            model, optimizer, criterion, loss = train_labeled(model, optimizer, criterion, train_loader_labeled, task, device)
            epoch, auc = val_labeled(dataset_name, model, optimizer, scheduler, val_loader_labeled, task, val_auc_list, dir_path, epoch, auc_old, epoch_old, loss, device)
        auc_list = np.array(val_auc_list)
        index = auc_list.argmax()
        print('epoch %s is the best model' % (index))

        # evaluate model
        saveBestModel(dir_path, model, train_loader_labeled, val_loader_labeled, test_loader_labeled, task, auc_list, index, inputs, device)
            
    
    elif task_input == "NoisyStudent":
        print("==> NoisyStudent-Training...")

    elif task_input == "MTSS":
        print("==> MTSS-Training...")

    elif task_input == "Pseudolabel":
        print("==> Pseudolabel-Training...")

        for i in range(count_students+1):
            start_epoch = 0
            vars()["pseudolabel" + str(i)] = PseudoLabel(inputs, device)
            if i == 0:
                model, optimizer, scheduler, criterion, model_dir = vars()["pseudolabel" + str(i)].defineOperators(n_channels, n_classes, dir_path, train_loader_unlabeled)
                if os.path.isdir(model_dir) and len(os.listdir(model_dir)) != 0:
                    list_of_files = glob.glob(model_dir + "/*.pth")
                    latest_file = max(list_of_files, key=os.path.getctime)
                    filename = latest_file
                    net_create = CreateModel(dataset_name, n_channels, n_classes, device)
                    model, optimizer, scheduler, loss, start_epoch, val_auc_list = net_create.load_checkpoint(model, optimizer, scheduler, filename)
                    
                else:
                    # start training
                    auc_return, epoch_return, auc_list, index = vars()["pseudolabel" + str(i)].startTraining(start_epoch, num_epoch, dir_path, train_loader_labeled, val_loader_labeled, model, criterion, optimizer, scheduler, task, val_auc_list, auc_old, epoch_old)
                
            else:
                # create pseudo labels
                pseudolabeled_dataloader = vars()["pseudolabel" + str(i)].create_pseudolabels(model, train_dataset_unlabeled, train_loader_unlabeled, batch_size, device)
                for index, (inputs, targets) in enumerate(pseudolabeled_dataloader):
                    print(inputs)
                    print(targets)
                
                # create operators
                vars()["net" + str(i)], vars()["optimizer" + str(i)], vars()["scheduler" + str(i)], vars()["criterion" + str(i)], model_dir = vars()["pseudolabel" + str(i)].defineOperators(n_channels, n_classes, dir_path, pseudolabeled_dataloader)
                # train model
                auc_return, epoch_return, auc_list, index = vars()["pseudolabel" + str(i)].startTraining(start_epoch, num_epoch, dir_path, pseudolabeled_dataloader, val_loader_labeled, model, criterion, optimizer, scheduler, task, val_auc_list, auc_old, epoch_old)
                # evaluate and save model
                vars()["pseudolabel" + str(i)].saveBestModel(dir_path, model, pseudolabeled_dataloader, val_loader_labeled, test_loader_labeled, task, auc_list, index)
            
        
    # # ****************************************** create student nets **********************************************************
    #     for i in range(count_students):
    #     # ******************** create net architectures ********************
    #         vars()["net_create" + str(i)] = CreateModel(dataset_name, n_channels, n_classes, device)
    #         vars()["studentnet_" + str(i)], vars()["image_size" + str(i)] = vars()["net_create" + str(i)].createNewCNN(net_input)
            
    #     # ******************** create loss function ************************
    #         vars()["lossfun" + str(i)] = LossFunction()
    #         vars()["criterion" + str(i)] = vars()["lossfun" + str(i)].createLossFunction(loss_function)
            
    #     # ******************** create optimizer ****************************
    #         vars()["student_optimizer" + str(i)] = Optimizer()
    #         vars()["optimizer" + str(i)] = vars()["student_optimizer" + str(i)].createOptimizer(optimizer_input,vars()["studentnet_" + str(i)], momentum, weight_decay, learning_rate)
    #         vars()["scheduler" + str(i)] = vars()["student_optimizer" + str(i)].createScheduler(vars()["optimizer" + str(i)], len(train_loader_labeled), milestone_count, decayLr)
        


                