import torch
from torch.nn import functional as F
import torch.nn as nn
import os
import glob
import shutil
import numpy as np
from shutil import copyfile
from tqdm import trange
from classes.CreateModel import CreateModel, LossFunction, Optimizer
from classes.PrepareData import createDataLoader
from classes.Operations import train_labeled, val_labeled, test_labeled

class PseudoLabel:
    def __init__(self, inputs, device):
        self.inputs = inputs
        self.device = device
        
    def defineOperators(self, n_channels, n_classes, dir_path, data_loader):
        # ************************************** create net architectures **************************************
        net_create = CreateModel(self.inputs['dataset'], n_channels, n_classes, self.device)

        # ************************************** create loss function ******************************************
        Lossfun = LossFunction()
        
        # ************************************** create optimizer **********************************************
        OptimizerInst = Optimizer()

        model, image_size = net_create.createNewCNN(self.inputs['net_input'])
        optimizer = OptimizerInst.createOptimizer(self.inputs['optimizer'], model, self.inputs['momentum'], self.inputs['weight_decay'], self.inputs['lr'])
        scheduler = OptimizerInst.createScheduler(optimizer, len(data_loader), self.inputs['milestone_count'], self.inputs['decayLr'])
        model_dir = os.path.join(self.inputs['output_root'], self.inputs['dataset'] + "_" + str(self.inputs['train_size']))
        criterion = Lossfun.createLossFunction(self.inputs['loss_function'])
        
        return model, optimizer, scheduler, criterion, model_dir


    def startTraining(self, start_epoch, num_epoch, dir_path, train_loader, val_loader, model, criterion, optimizer, scheduler, task, val_auc_list, auc_old, epoch_old):

        # start training
        for epoch in trange(start_epoch, num_epoch):
            model, optimizer, criterion, loss = train_labeled(model, optimizer, criterion, train_loader, task, self.device)
            epoch_return, auc_return = val_labeled(self.inputs['dataset'], model, optimizer, scheduler, val_loader, task, val_auc_list, dir_path, epoch, auc_old, epoch_old, loss, self.device)

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


    # ****************************************** start Pseudolabeling *********************************************************
    def create_pseudolabels(self, model, dataset, loader, batch_size, device):
        model.eval()
        results = np.zeros((len(dataset), 10), dtype=np.float32)

        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs = inputs.to(device)

            outputs = model(inputs)
            prob = F.softmax(outputs, dim=1)
            
            #results = prob.cpu().detach().numpy().tolist()
            results = torch.max(prob.cpu().detach(),1)
            print("inputs: ", inputs)
            print("targets", targets)
            print("result1:", results[0])
            new_dataset = torch.utils.data.TensorDataset(inputs, results[0])
            student_dataloader = createDataLoader(new_dataset, batch_size)
        return student_dataloader


    def saveBestModel(self, dir_path, model, train_loader, val_loader, test_loader, task, auc_list, index):
        #*************************** Evaluate trained Model **************************************
        print('==> Testing model...')
        # restore_model_path = os.path.join(
        #     dir_path, 'ckpt_%d_auc_%.5f.pth' % (index, auc_list[index]))
        restore_model_path = os.path.join(dir_path,'training_'+ self.inputs['dataset_name'])
        
        model.load_state_dict(torch.load(restore_model_path)['net'])
        test_labeled('train', model, train_loader, self.inputs['dataset'], task, device, output_root=self.inputs['output_root'])
        test_labeled('val', model, val_loader, self.inputs['dataset'], task, device, output_root=self.inputs['output_root'])
        test_labeled('test', model, test_loader, self.inputs['dataset'], task, device, output_root=self.inputs['output_root'])
            
        save_best_model_path = os.path.join( 
            os.path.join(self.inputs['output_root'], self.inputs['dataset']), 'ckpt_%d_auc_%.5f.pth' % (index, auc_list[index]))
        copyfile(restore_model_path, save_best_model_path)
        shutil.rmtree(dir_path)