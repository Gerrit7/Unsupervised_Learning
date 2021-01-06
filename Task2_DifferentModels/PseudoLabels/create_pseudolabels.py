import os
import argparse
from tqdm import trange
import numpy as np
import shutil
from shutil import copyfile
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import csv 
import glob

from medmnist.models import ResNet18, ResNet50
from medmnist.dataset import PathMNIST, ChestMNIST, DermaMNIST, OCTMNIST, PneumoniaMNIST, RetinaMNIST, \
    BreastMNIST, OrganMNISTAxial, OrganMNISTCoronal, OrganMNISTSagittal
from medmnist.evaluator import getAUC, getACC, save_results
from medmnist.info import INFO
from model.inception_resnet_v2 import Inception_ResNetv2

def main():
    # check if cuda is available for training on cpu
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

    # loading trainer net for pseudo-labeling unlabeled data
    print(dir_path)
    list_of_files = glob.glob(dir_path + "/*")
    latest_file = max(list_of_files, key=os.path.getctime)
    filename = latest_file
    print(filename)
    model, optimizer, start_epoch, val_auc_list = load_checkpoint(model, optimizer, val_auc_list, filename)
    # now individually transfer the optimizer parts...
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    model.to(device)
    return 0

def load_checkpoint(model, optimizer, val_auc_list, filename='checkpoint.pth.tar'):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        print("loading epoch")
        start_epoch = checkpoint['epoch']
        print("loading model")
        model.load_state_dict(checkpoint['net'])
        print("loading optimizer")
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("loading auc_list")
        val_auc_list = checkpoint['auc']
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, start_epoch, val_auc_list

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='RUN Baseline model of MedMNIST')
    parser.add_argument('--data_name',
                        default='pathmnist',
                        help='subset of MedMNIST',
                        type=str)
    parser.add_argument('--input_root',
                        default='./input',
                        help='input root, the source of dataset files',
                        type=str)
    parser.add_argument('--output_root',
                        default='./output',
                        help='output root, where to save models and results',
                        type=str)
    parser.add_argument('--num_epoch',
                        default=100,
                        help='num of epochs of training',
                        type=int)
    parser.add_argument('--train_size',
                        default=1.0,
                        help='size of trainingdata',
                        type=float)
    parser.add_argument('--download',
                        help='whether download the dataset or not',
                        default=True,
                        action='store_true')
    parser.add_argument('--continue_train',
                        help='continue training?',
                        default=True,
                        action='store_false')

    args = parser.parse_args()
    data_name = args.data_name.lower()
    input_root = args.input_root
    output_root = args.output_root
    end_epoch = args.num_epoch
    train_size = args.train_size
    download = args.download
    continue_train = args.continue_train
    print(download)
    main(data_name,
         input_root,
         output_root,
         end_epoch = end_epoch,
         trainSize = train_size,
         download = download,
         continueTrain = continue_train)

