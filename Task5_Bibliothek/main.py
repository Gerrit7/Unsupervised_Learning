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
import torch
from torch.nn import functional as F
import torch.nn as nn
from tqdm import trange
import torch.utils.data as data

from models.models import EfficientNet, ResNet18, ResNet50
from classes.CreateModel import createNewCNN, load_checkpoint, createLossFunction, createOptimizer, createScheduler
from classes.Operations import train_labeled, val_labeled, test_labeled, evalModel, defineOperators, create_pseudolabels, predict
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
    
    train_dataset, train_subset_labeled, train_dataset_labeled, train_subset_unlabeled, train_dataset_unlabeled = prepareClass.prepareDataSet('train', train_transform, train_size)
    train_loader_labeled   = createDataLoader(train_subset_labeled, batch_size)
    if len(train_dataset_unlabeled)>0:
        train_loader_unlabeled = createDataLoader(train_subset_unlabeled, batch_size)
    #print(len(train_loader_labeled), len(train_loader_unlabeled))
    
    val_dataset, val_subset_labeled, val_dataset_labeled, val_subset_unlabeled, val_dataset_unlabeled = prepareClass.prepareDataSet('val', val_transform, 1)
    val_loader_labeled   = createDataLoader(val_subset_labeled, batch_size)
    #val_loader_unlabeled = createDataLoader(val_subset_unlabeled, batch_size)

    test_dataset, test_subset_labeled, test_dataset_labeled, test_subset_unlabeled, test_dataset_unlabeled = prepareClass.prepareDataSet('test', test_transform, 1)
    test_loader_labeled   = createDataLoader(test_subset_labeled, batch_size)
    #test_loader_unlabeled = createDataLoader(test_subset_unlabeled, batch_size)
    
    print('Train: ', len(train_subset_labeled), ', Valid: ', len(val_subset_labeled), ', Test: ', len(test_subset_labeled))



# ************************************** Training  ***************************************************************************************************
    if task_input == "BaseLine":
        print("==> Baseline-Training...")

        # creating model
        start_epoch, model, criterion, optimizer, scheduler, epoch_old, auc_old = defineOperators(n_channels, n_classes, train_loader_labeled, dir_path, inputs, device)
        # training model
        for epoch in trange(start_epoch, num_epoch):
            model, optimizer, criterion, loss = train_labeled(model, optimizer, criterion, train_loader_labeled, task, device)
            epoch, auc = val_labeled(model, optimizer, scheduler, val_loader_labeled, task, val_auc_list, dir_path, epoch, auc_old, epoch_old, loss, device)
            auc_old = auc
            epoch_old = epoch
        auc_list = np.array(val_auc_list)
        index = auc_list.argmax()
        print('epoch %s is the best model' % (index))
        
        savepath = os.path.join(dir_path, inputs['dataset'] + '%.2f' % train_size + "_bestmodel.pth")
        torch.save(model.state_dict(), savepath)
        # evaluate model
        evalModel(dir_path, model, train_loader_labeled, val_loader_labeled, test_loader_labeled, task, auc_list, index, inputs, device)
            
    
    elif task_input == "NoisyStudent":
        print("==> NoisyStudent-Training...")

    elif task_input == "MTSS":
        print("==> MTSS-Training...")

# ************************************** PseudoLabeling  *************************************************************************************************
    elif task_input == "Pseudolabel":
        print("==> Pseudolabel-Training...")

        labeled_inputs = []
        labeled_targets = []  
        for batch_idx, (train_inputs, train_targets) in enumerate(train_loader_labeled):
            for image in train_inputs:
                labeled_inputs.append(image.reshape(28,28))
            for tupl in train_targets:
                labeled_targets.append(np.argmax(tupl))
            
        labeled_inputs = torch.stack(labeled_inputs)
        labeled_targets = torch.stack(labeled_targets)

        # ************************************** Create Models  *****************************************
        # create teacher model
        model, image_size = createNewCNN(inputs['net_input'], n_channels, n_classes, device)
        
        # load model parameters
        if os.path.isdir(dir_path) and len(os.listdir(dir_path)) != 0:
            list_of_files = glob.glob(dir_path + "/*bestmodel.pth")
            latest_file = max(list_of_files, key=os.path.getctime)
            model.load_state_dict(torch.load(Path(latest_file)))
            #model, optimizer, scheduler, loss, start_epoch, val_auc_list = load_checkpoint(model, optimizer, scheduler, latest_file)

        combined_dataloader = train_loader_unlabeled
        for i in range(count_students):
            val_auc_list = []
            epoch_old = 0
            auc_old = 0
            loss = 0
            start_epoch = 0
            # start pseudolabeling with teacher model
            print("==> PseudoLabeling...")
            
            #test_labeled('train', model, train_loader_labeled, flag, train_size, task, device, output_root=None)
            
            #pseudo_inputs, pseudo_targets = create_pseudolabels(model, train_dataset_unlabeled, train_loader_unlabeled, batch_size, task, device)
            pseudo_inputs, pseudo_targets = predict(combined_dataloader, model, device)

            print(type(pseudo_inputs))
            print(pseudo_inputs.size())
            print(type(pseudo_targets))
            print(pseudo_targets.size())

            print(type(labeled_inputs))
            print(labeled_inputs.size())
            print(type(labeled_targets))
            print(labeled_targets.size())
            combined_inputs = np.concatenate((np.asarray(labeled_inputs), np.asarray(pseudo_inputs))) 
            combined_targets = np.concatenate((np.asarray(labeled_targets), np.asarray(pseudo_targets)))
            #np.savez(os.path.join(dir_path, 'combined_' + inputs['dataset'] +'.npz'), inputs=combined_inputs, targets=combined_targets)
            combined_inputs = torch.from_numpy(combined_inputs)
            combined_targets = torch.from_numpy(combined_targets)

            pseudo_dataset = data.TensorDataset(combined_inputs, combined_targets)
            combined_dataloader = data.DataLoader(dataset=pseudo_dataset, batch_size=batch_size, shuffle=True)

            # create student model
            model, image_size = createNewCNN(inputs['net_input'], n_channels, n_classes, device)
            criterion = createLossFunction(inputs['loss_function'])
            optimizer = createOptimizer(inputs['optimizer'], model, inputs['momentum'], inputs['weight_decay'], inputs['lr'])
            scheduler = createScheduler(optimizer, len(combined_dataloader), inputs['milestone_count'], inputs['decayLr'])

            #training model
            student_path = os.path.join(dir_path, "training_"+ str(i))
            os.mkdir(student_path)
            for epoch in trange(start_epoch, num_epoch):
                model, optimizer, criterion, loss = train_labeled(model, optimizer, criterion, train_loader_labeled, task, device)
                epoch, auc = val_labeled(model, optimizer, scheduler, val_loader_labeled, task, val_auc_list, student_path, epoch, auc_old, epoch_old, loss, device)
                auc_old = auc
                epoch_old = epoch
            auc_list = np.array(val_auc_list)
            index = auc_list.argmax()
            print('epoch %s is the best model' % (index))
            
            # evaluate model
            evalModel(student_path, model, train_loader_labeled, val_loader_labeled, test_loader_labeled, task, auc_list, index, inputs, device)
            

        # for i in range(count_students+1):
        #     val_auc_list = []
        #     epoch_old = 0
        #     auc_old = 0
        #     loss = 0
        #     start_epoch = 0
        #     if i == 0: # first train teacher model

        #         if os.path.isdir(dir_path) and len(os.listdir(dir_path)) != 0:
        #             list_of_files = glob.glob(dir_path + "/*.pth")
        #             latest_file = max(list_of_files, key=os.path.getctime)
        #             filename = latest_file
        #             net_create = CreateModel(n_channels, n_classes, device)
        #             start_epoch, model, criterion, optimizer, scheduler, epoch_old, auc_old = defineOperators(n_channels, n_classes, train_loader_labeled, dir_path, inputs, device)
        #             model, optimizer, scheduler, loss, start_epoch, val_auc_list = net_create.load_checkpoint(model, optimizer, scheduler, filename)
                    
        #         else:
        #             # creating model
        #             start_epoch, model, criterion, optimizer, scheduler, epoch_old, auc_old = defineOperators(n_channels, n_classes, train_loader_labeled, dir_path, inputs, device)
        #             # training model
        #             print("==> Training teacher model...")
        #             for epoch in trange(start_epoch, num_epoch):
        #                 model, optimizer, criterion, loss = train_labeled(model, optimizer, criterion, train_loader_labeled, task, device)
        #                 epoch, auc = val_labeled(model, optimizer, scheduler, val_loader_labeled, task, val_auc_list, dir_path, epoch, auc_old, epoch_old, loss, device)
        #                 auc_old = auc
        #                 epoch_old = epoch
        #             auc_list = np.array(val_auc_list)
        #             index = auc_list.argmax()
        #             print('epoch %s is the best model' % (index))

        #             # evaluate model
        #             saveBestModel(dir_path, model, train_loader_labeled, val_loader_labeled, test_loader_labeled, task, auc_list, index, inputs, device)
            
        #     else:
        #         student_path = os.path.join(dir_path, "training_"+ str(i))
        #         # create pseudo labels
        #         pseudolabeled_dataloader = create_pseudolabels(model, train_dataset_unlabeled, train_loader_unlabeled, batch_size, device)
                 
        #         if train_teacher_dependent == False:
        #             # create operators
        #             start_epoch, model, criterion, optimizer, scheduler, epoch_old, auc_old = defineOperators(n_channels, n_classes, pseudolabeled_dataloader, student_path, inputs, device)
               
        #             # training model
        #             print("==> Training student on labeled Data...")
        #             for epoch in trange(start_epoch, num_epoch):
        #                 model, optimizer, criterion, loss = train_labeled(model, optimizer, criterion, train_loader_labeled, task, device)
        #                 epoch, auc = val_labeled(model, optimizer, scheduler, val_loader_labeled, task, val_auc_list, student_path, epoch, auc_old, epoch_old, loss, device)
        #                 auc_old = auc
        #                 epoch_old = epoch

        #         val_auc_list = []
        #         epoch_old = 0
        #         auc_old = 0
        #         loss = 0
        #         start_epoch = 0
        #         print("==> Training student on pseudolabeled Data...")
        #         for epoch in trange(start_epoch, num_epoch):
        #             model, optimizer, criterion, loss = train_labeled(model, optimizer, criterion, pseudolabeled_dataloader, task, device)
        #             epoch, auc = val_labeled(model, optimizer, scheduler, val_loader_labeled, task, val_auc_list, student_path, epoch, auc_old, epoch_old, loss, device)
        #             auc_old = auc
        #             epoch_old = epoch
        #         auc_list = np.array(val_auc_list)
        #         index = auc_list.argmax()
        #         print('epoch %s is the best model' % (index))

        #         # evaluate model
        #         saveBestModel(student_path, model, train_loader_labeled, val_loader_labeled, test_loader_labeled, task, auc_list, index, inputs, device)
    
    
                