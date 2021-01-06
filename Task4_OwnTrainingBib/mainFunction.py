import tkinter as tk
from tkinter import *
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
import numpy as np
import shutil
from shutil import copyfile
import os
import glob

# import different model structures
from models.models import EfficientNet, ResNet18, ResNet50
from models.models import Inception_ResNetv2
# import dataset functions
from datasets.prepareData import prepareMedmnist
from datasets.medmnist.info import INFO
# import net functions
from NetFunctions.NetFunctions import  train, test, val, createPseudoLabel, load_checkpoint


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
         mode,
         task_input,
         optimizer,
         decayLr,
         milestone_count,
         loss_function,
         augmentations,
         download):
    
    # print(dataset_name)
    # print(data_root)
    # print(output_root)
    # print(num_epoch)
    # print(batch_size)
    # print(learning_rate)
    # print(momentum)
    # print(train_size)
    # print(weight_decay)
    # print(net_input)
    # print(mode)
    # print(task_input)
    # print(optimizer)
    # print(augmentations)
    # print(download)
    
    start_epoch = 0
    # Setting information depending on selected dataset 
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


    # Setting device for training 
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

        
    if mode.get() == False: 
        mode = "training"
    else:
        mode = "prediction"
    
    # creating net and prepare data for usage with this net
    if net_input == 'Resnet18':
        model = ResNet18(in_channels=n_channels, num_classes=n_classes).to(device)
        image_size = 28
        train_loader, train_loader_unused_data, val_loader, test_loader = prepareMedmnist(flag, data_root, output_root, net_input, image_size, augmentations, batch_size, train_size, download)
        print('using ResNet18')
    
    elif net_input == 'Resnet50':
        model = ResNet50(in_channels=n_channels, num_classes=n_classes).to(device)
        image_size = 28
        train_loader, train_loader_unused_data, val_loader, test_loader = prepareMedmnist(flag, data_root, output_root, net_input, image_size, augmentations, batch_size, train_size, download)
        print('using ResNet50')
    
    elif net_input == 'EfficientNet-b0':
        model = EfficientNet.from_name('efficientnet-b0',n_channels)
        model._fc= nn.Linear(1280, n_classes)
        model.to(device)
        image_size = 224
        train_loader, train_loader_unused_data, val_loader, test_loader = prepareMedmnist(flag, data_root, output_root, net_input, image_size, augmentations, batch_size, train_size, download)
        print('using EfficientNet-b0')
    
    elif net_input == 'EfficientNet-b1':
        model = EfficientNet.from_name('efficientnet-b1',n_channels)
        model._fc= nn.Linear(1280, n_classes)
        model.to(device)
        image_size = 224
        train_loader, train_loader_unused_data, val_loader, test_loader = prepareMedmnist(flag, data_root, output_root, net_input, image_size, augmentations, batch_size, train_size, download)
        print('using EfficientNet-b1')
    
    elif net_input == 'EfficientNet-b7':
        model = EfficientNet.from_name('efficientnet-b7',n_channels)
        model._fc= nn.Linear(1280, n_classes)
        model.to(device)
        image_size = 224
        train_loader, train_loader_unused_data, val_loader, test_loader = prepareMedmnist(flag, data_root, output_root, net_input, image_size, augmentations, batch_size, train_size, download)
        print('using EfficientNet-b7')
    
    elif net_input == 'Inception-Resnet-V2':
        model = Inception_ResNetv2(n_channels)
        model.linear= nn.Linear(1536, n_classes, True)
        image_size = 256
        train_loader, train_loader_unused_data, val_loader, test_loader = prepareMedmnist(flag, data_root, output_root, net_input, image_size, augmentations, batch_size, train_size, download)
        print('using Inception-Resnet-V2')

    else:
        print("Net not found!")

    # setting up the loss function
    if loss_function == "crossentropyloss":
        criterion = nn.CrossEntropyLoss()
    elif loss_function == "bce":
        criterion = nn.BCEWithLogitsLoss()
    elif loss_function == "MSE":
        criterion == nn.MSELoss()
    elif loss_function == "MLE":
        print("actual not supported!")
    else:
        print("failure: using default loss function CE")
        criterion = nn.CrossEntropyLoss()

    # setting up the optimizer
    if optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        print('optimizer is SGD')
    elif optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        print('optimizer is Adam')
    elif optimizer == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum, centered=False)
        print('optimizer is RMSprop')
    else:
        print("undefined optimizer: taking default SGD")
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    
    if task_input == "BaseLine":
        print("creating Baseline-Training")
        # check wheather a training session has already startet for this dataset
        if os.path.isdir(dir_path) and len(os.listdir(dir_path)) != 0:
            # loading old checkpoint for further training
            list_of_files = glob.glob(dir_path + "/*")
            latest_file = max(list_of_files, key=os.path.getctime)
            filename = latest_file
            model, optimizer, loss, start_epoch, val_auc_list = load_checkpoint(model, optimizer, val_auc_list, filename)
            # now individually transfer the optimizer parts...
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            model.to(device)
        else:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
        
        # use Learning rate decay 
        if milestone_count > 0:
            milestones = []
            for i in range(len(milestone_count)):
                milestones.append(len(train_loader)/milestone_count*i)
            optimizer = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=decayLr)

        # start training
        for epoch in trange(start_epoch, num_epoch):
            optimizer, loss = train(model, optimizer, criterion, train_loader, device, task)
            epoch_return, auc_return = val(model, val_loader, device, val_auc_list, task, dir_path, epoch, auc_old, epoch_old, optimizer, loss)
            if auc_return > auc_old:
                epoch_old = epoch_return
                auc_old = auc_return
            
        auc_list = np.array(val_auc_list)
        index = auc_list.argmax()
        print('epoch %s is the best model' % (index))


        # testing train, validation and test dataset with best model parameters
        print('==> Testing model...')
        restore_model_path = os.path.join(
            dir_path, 'ckpt_%d_auc_%.5f.pth' % (index, auc_list[index]))

        model.load_state_dict(torch.load(restore_model_path)['net'])
        test(model, 'train', train_loader, device, dataset_name, task, output_root=output_root)
        test(model, 'val', val_loader, device, dataset_name, task, output_root=output_root)
        test(model,'test', test_loader, device, dataset_name, task, output_root=output_root)
            
        save_best_model_path = os.path.join( 
            os.path.join(output_root, dataset_name), 'ckpt_%d_auc_%.5f.pth' % (index, auc_list[index]))
        copyfile(restore_model_path, save_best_model_path)
        shutil.rmtree(dir_path)

    elif task_input == "NoisyStudent":
        print("creating NoisyStudent")

    elif task_input == "MTSS":
        print("training multiple teachers and combine outputs")

    elif task_input == "Pseudolabel":
        print("creating pseudo labels for unlabeled dataset")
        os.chdir(os.path.join(output_root, dataset_name))
        restore_model_path = glob.glob("*.pth")[0]
        model, optimizer, loss, start_epoch, val_auc_list = load_checkpoint(model, optimizer, val_auc_list, restore_model_path)
        
        createPseudoLabel(model, train_loader_unused_data, device, task, output_root=None)
    else:
        print("task not found!")
        



    