import torch
import torch.nn as nn
import torch.optim as optim
import os
# import different model structures
from models.models import EfficientNet, ResNet18, ResNet50
from models.models import Inception_ResNetv2

class CreateModel():
    def __init__(self, dataset_name, n_channels, n_classes, device):
        self. dataset_name = dataset_name
        self.n_channels = n_channels
        self. n_classes = n_classes
        self.device = device

    def createNewCNN(self, net_input):
        # creating net and prepare data for usage with this net
        if net_input == 'Resnet18':
            model = ResNet18(in_channels=self.n_channels, num_classes=self.n_classes).to(self.device)
            image_size = 28
            print('using ResNet18')
        
        elif net_input == 'Resnet50':
            model = ResNet50(in_channels=self.n_channels, num_classes=self.n_classes).to(self.device)
            image_size = 28
            print('using ResNet50')
        
        elif net_input == 'EfficientNet-b0':
            model = EfficientNet.from_name('efficientnet-b0',self.n_channels)
            model._fc= nn.Linear(1280, self.n_classes)
            model.to(self.device)
            image_size = 224
            print('using EfficientNet-b0')
        
        elif net_input == 'EfficientNet-b1':
            model = EfficientNet.from_name('efficientnet-b1',self.n_channels)
            model._fc= nn.Linear(1280, self.n_classes)
            model.to(self.device)
            image_size = 224
            print('using EfficientNet-b1')
        
        elif net_input == 'EfficientNet-b7':
            model = EfficientNet.from_name('efficientnet-b7',self.n_channels)
            model._fc= nn.Linear(1280, self.n_classes)
            model.to(self.device)
            image_size = 224
            print('using EfficientNet-b7')
        
        elif net_input == 'Inception-Resnet-V2':
            model = Inception_ResNetv2(self.n_channels)
            model.linear= nn.Linear(1536, self.n_classes, True)
            image_size = 256
            print('using Inception-Resnet-V2')

        else:
            print("Net not found!")

        return model, image_size

    def load_checkpoint(self, model, optimizer, val_auc_list, filename='checkpoint.pth.tar'):
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



class LossFunction():
    def createLossFunction(self, loss_function):
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
        return criterion

class Optimizer():
    def createOptimizer(self, optimizer_input, model, momentum, weight_decay, learning_rate):
        # setting up the optimizer
        if optimizer_input == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
            print('optimizer is SGD')
        elif optimizer_input == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            print('optimizer is Adam')
        elif optimizer_input == 'RMSprop':
            optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum, centered=False)
            print('optimizer is RMSprop')
        else:
            print("undefined optimizer: taking default SGD")
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        return optimizer

    def createScheduler(self, optimizer, size_trainloader, milestone_count, decayLr):
        milestones = []
        for i in range(milestone_count):
            milestones.append(int(round(size_trainloader/milestone_count*i)))
            print(milestones)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=decayLr)
        return scheduler
