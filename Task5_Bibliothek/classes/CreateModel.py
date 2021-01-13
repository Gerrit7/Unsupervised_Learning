import torch
import torch.nn as nn
import torch.optim as optim
import os
# import different model structures
from models.models import EfficientNet, ResNet18, ResNet50
from models.models import Inception_ResNetv2

def createNewCNN(net_input, n_channels, n_classes, device):
    # creating net and prepare data for usage with this net
    if net_input == 'Resnet18':
        model = ResNet18(in_channels=n_channels, num_classes=n_classes).to(device)
        image_size = 28
        #print('using ResNet18')
    
    elif net_input == 'Resnet50':
        model = ResNet50(in_channels=n_channels, num_classes=n_classes).to(device)
        image_size = 28
        #print('using ResNet50')
    
    elif net_input == 'EfficientNet-b0':
        model = EfficientNet.from_name('efficientnet-b0',n_channels)
        model._fc= nn.Linear(1280, n_classes)
        model.to(device)
        image_size = 224
        #print('using EfficientNet-b0')
    
    elif net_input == 'EfficientNet-b1':
        model = EfficientNet.from_name('efficientnet-b1',n_channels)
        model._fc= nn.Linear(1280, n_classes)
        model.to(device)
        image_size = 224
        #print('using EfficientNet-b1')
    
    elif net_input == 'EfficientNet-b7':
        model = EfficientNet.from_name('efficientnet-b7',n_channels)
        model._fc= nn.Linear(1280, n_classes)
        model.to(device)
        image_size = 224
        #print('using EfficientNet-b7')
    
    elif net_input == 'Inception-Resnet-V2':
        model = Inception_ResNetv2(n_channels)
        model.linear= nn.Linear(1536, n_classes, True)
        image_size = 256
        #print('using Inception-Resnet-V2')

    else:
        print("Net not found!")

    return model, image_size

def load_checkpoint(model, optimizer, scheduler, filename='checkpoint.pth.tar'):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)


        model.load_state_dict(checkpoint['net'])
        val_auc_list = checkpoint['auc_list']
        start_epoch = checkpoint['epoch']+1
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        loss = checkpoint['loss']

        
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(filename, checkpoint['epoch']))
        return model, optimizer, scheduler, loss, start_epoch, val_auc_list
    else:
        print("=> no checkpoint found at '{}'".format(filename))

        

def createLossFunction(loss_function):
    # setting up the loss function
    if loss_function == "crossentropyloss":
        criterion = nn.CrossEntropyLoss()
    elif loss_function == "bce":
        criterion = nn.BCEWithLogitsLoss()
    elif loss_function == "MSE":
        criterion = nn.MSELoss()
    elif loss_function == "MLE":
        print("actual not supported!")
    else:
        print("failure: using default loss function CE")
        criterion = nn.CrossEntropyLoss()
    return criterion


def createOptimizer(optimizer_input, model, momentum, weight_decay, learning_rate):
    # setting up the optimizer
    if optimizer_input == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        #print('optimizer is SGD')
    elif optimizer_input == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        #print('optimizer is Adam')
    elif optimizer_input == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum, centered=False)
        #print('optimizer is RMSprop')
    else:
        print("undefined optimizer: taking default SGD")
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    return optimizer

def createScheduler(optimizer, size_trainloader, milestone_count, decayLr):
    milestones = []
    for i in range(milestone_count):
        milestones.append(int(round(size_trainloader/milestone_count*i)))
        #print(milestones)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=decayLr)
    return scheduler
