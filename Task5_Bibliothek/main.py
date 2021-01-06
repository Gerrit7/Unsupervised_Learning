import glob
import os
import shutil
from shutil import copyfile

import numpy as np
import torch
from tqdm import trange

from classes.CreateModel import CreateModel, LossFunction, Optimizer
from classes.Operations import SupervisedLearning
from classes.PrepareData import PrepareData
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
         mode,
         task_input,
         optimizer,
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
        'mode': mode,
        'task': task_input,
        'optimizer': optimizer,
        'augs': augmentations,
        'download': download
    }

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
    
    train_loader_labeled, train_loader_unlabeled = prepareClass.createDataLoader('train', train_transform, batch_size, train_size)
    print(len(train_loader_labeled), len(train_loader_unlabeled))
    val_loader_labeled, val_loader_unlabeled = prepareClass.createDataLoader('train', train_transform, batch_size, train_size)
    test_loader_labeled, test_loader_unlabeled = prepareClass.createDataLoader('train', train_transform, batch_size, train_size)
    
    # ************************************** create net architectures **************************************
    teachernet = CreateModel(dataset_name, n_channels, n_classes, device)
    
    # ************************************** create loss function ******************************************
    teacherlossfun = LossFunction()
    
    # ************************************** create optimizer **********************************************
    teacheroptimizer = Optimizer()


    # ************************************** Training  *****************************************************
    if task_input == "BaseLine":
        print("==> Baseline-Training...")
        
        model, image_size = teachernet.createNewCNN(net_input)
        criterion = teacherlossfun.createLossFunction(loss_function)
        optimizer = teacheroptimizer.createOptimizer(optimizer,model, momentum, weight_decay, learning_rate)
        scheduler = teacheroptimizer.createScheduler(optimizer, len(train_loader_labeled), milestone_count, decayLr)

        # check wheather a training session has already startet for this dataset
        if os.path.isdir(dir_path) and len(os.listdir(dir_path)) != 0:
            list_of_files = glob.glob(dir_path + "/*")
            latest_file = max(list_of_files, key=os.path.getctime)
            filename = latest_file
            model, optimizer, loss, start_epoch, val_auc_list = teachernet.load_checkpoint(model, optimizer, val_auc_list, filename)
            
        else:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            
        # push optimizer and model to device (cpu/gpu)
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        model.to(device)

        # create Operations instance
        supervisedlearning = SupervisedLearning(model, optimizer, scheduler, criterion, device)
        
        print(optimizer)
        print(criterion)
        
        print(len(train_loader_labeled))
        print(train_loader_labeled)
        print(model)
        # start training
        for epoch in trange(start_epoch, num_epoch):
            optimizer, loss = supervisedlearning.train_labeled(train_loader_labeled, task)
            epoch_return, auc_return = supervisedlearning.val_labeled(val_loader_labeled, val_auc_list, task, dir_path, epoch, auc_old, epoch_old, loss)
            if auc_return > auc_old:
                epoch_old = epoch_return
                auc_old = auc_return
            scheduler.step()
            #print("epoch: ", epoch, ", old auc: ", auc_old, ", new auc: ", auc_return)
            
        auc_list = np.array(val_auc_list)
        index = auc_list.argmax()
        print('epoch %s is the best model' % (index))

    elif task_input == "NoisyStudent":
        print("==> NoisyStudent-Training...")

    elif task_input == "MTSS":
        print("==> MTSS-Training...")

    elif task_input == "Pseudolabel":
        print("==> Pseudolabel-Training...")

    # ************************************** Testing *******************************************************
    
    print('==> Testing model...')
    restore_model_path = os.path.join(
        dir_path, 'ckpt_%d_auc_%.5f.pth' % (index, auc_list[index]))

    model.load_state_dict(torch.load(restore_model_path)['net'])
    supervisedlearning.test_labeled('train', train_loader_labeled, dataset_name, task, output_root=output_root)
    supervisedlearning.test_labeled('val', val_loader_labeled, dataset_name, task, output_root=output_root)
    supervisedlearning.test_labeled('test', test_loader_labeled, dataset_name, task, output_root=output_root)
        
    save_best_model_path = os.path.join( 
        os.path.join(output_root, dataset_name), 'ckpt_%d_auc_%.5f.pth' % (index, auc_list[index]))
    copyfile(restore_model_path, save_best_model_path)
    shutil.rmtree(dir_path)

    
