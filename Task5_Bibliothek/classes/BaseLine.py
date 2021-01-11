import torch
import os
import glob
import numpy as np
import shutil
from shutil import copyfile
from tqdm import trange
from classes.CreateModel import CreateModel, LossFunction, Optimizer
from classes.Operations import train_labeled, val_labeled, test_labeled


def defineOperators(n_channels, n_classes, dataloader, dir_path, inputs, device):
    # ************************************** create net architectures **************************************
    net_create = CreateModel(inputs['dataset'], n_channels, n_classes, device)

    # ************************************** create loss function ******************************************
    teacherlossfun = LossFunction()
    
    # ************************************** create optimizer **********************************************
    teacheroptimizer = Optimizer()

    model, image_size = net_create.createNewCNN(inputs['net_input'])
    criterion = teacherlossfun.createLossFunction(inputs['loss_function'])
    optimizer = teacheroptimizer.createOptimizer(inputs['optimizer'], model, inputs['momentum'], inputs['weight_decay'], inputs['lr'])
    scheduler = teacheroptimizer.createScheduler(optimizer, len(dataloader), inputs['milestone_count'], inputs['decayLr'])

    # check wheather a training session has already startet for this dataset
    if os.path.isdir(dir_path) and len(os.listdir(dir_path)) != 0:
        list_of_files = glob.glob(dir_path + "/*")
        latest_file = max(list_of_files, key=os.path.getctime)
        filename = latest_file
        model, optimizer, scheduler, loss, start_epoch, val_auc_list = net_create.load_checkpoint(model, optimizer, scheduler, filename)
        epoch_old = start_epoch-1
        auc_old = val_auc_list[-1]
    else:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        start_epoch = 0
        epoch_old = 0
        auc_old = 0

    # push optimizer and model to device (cpu/gpu)
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    model.to(device)

    return start_epoch, model, criterion, optimizer, scheduler, epoch_old, auc_old

def startTraining(start_epoch, num_epoch, dir_path, train_loader, val_loader, model, criterion, optimizer, scheduler, task, val_auc_list, auc_old, epoch_old, inputs, device):

    # start training
    for epoch in trange(start_epoch, num_epoch):
        model, optimizer, criterion, loss = train_labeled(model, optimizer, criterion, train_loader, task, device)
        epoch_return, auc_return = val_labeled(inputs['dataset'], model, optimizer, scheduler, val_loader, task, val_auc_list, dir_path, epoch, auc_old, epoch_old, loss, device)

        #train(model, optimizer, criterion, train_loader_labeled, device, task)
        #val(model, val_loader_labeled, device, val_auc_list, task, dir_path, epoch)
        scheduler.step()

        if auc_return > auc_old:
            epoch_old = epoch_return
            auc_old = auc_return
        

    auc_list = np.array(val_auc_list)
    index = auc_list.argmax()
    print('epoch %s is the best model' % (index))

    return auc_return, epoch_return, auc_list, index


def saveBestModel(dir_path, model, train_loader, val_loader, test_loader, task, auc_list, index, inputs, device):
    #*************************** Evaluate trained Model **************************************
    print('==> Testing model...')
    restore_model_path = os.path.join(
        dir_path, 'ckpt_%d_auc_%.5f.pth' % (index, auc_list[index]))
    #restore_model_path = os.path.join(dir_path,)
    
    model.load_state_dict(torch.load(restore_model_path)['net'])
    test_labeled('train', model, train_loader, inputs['dataset'], inputs['train_size'], task, device, output_root=inputs['output_root'])
    test_labeled('val', model, val_loader, inputs['dataset'],  inputs['train_size'], task, device, output_root=inputs['output_root'])
    test_labeled('test', model, test_loader, inputs['dataset'],  inputs['train_size'], task, device, output_root=inputs['output_root'])
        
    save_best_model_path = os.path.join( 
        os.path.join(inputs['output_root'], inputs['dataset'] + "_" + str(inputs['train_size'])), 'ckpt_%d_auc_%.5f.pth' % (index, auc_list[index]))
    copyfile(restore_model_path, save_best_model_path)
    shutil.rmtree(dir_path)

