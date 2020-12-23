import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
import numpy as np
import shutil
from shutil import copyfile
import os

from models.models import EfficientNet, ResNet18, ResNet50

from datasets.prepareData import prepareMedmnist
from datasets.medmnist.info import INFO



from NetFunctions.NetFunctions import  train, test, val



def main(data_name,
         data_root,
         output_root,
         num_epoch,
         batch_size,
         learning_rate,
         momentum,
         train_size,
         weight_decay,
         model_input,
         optimizer,
         loss_function,
         augmentations,
         download):

    print(data_name)
    print(data_root)
    print(output_root)
    print(num_epoch)
    print(batch_size)
    print(learning_rate)
    print(momentum)
    print(train_size)
    print(weight_decay)
    print(model_input)
    print(optimizer)
    print(augmentations)
    print(download)
    
    if data_name == 'medmnist':
        while True:
            try:
                print()
                dataset_name = input('Specify subset of medmnsit dataset! (Type help for information)')

                if data_name == 'help':
                    print('Possible keywords: breastmnist, chestmnist, dermamnist, octmnist, organmnist_axial, ' +
                    'organmnist_coronal, organmnist_sagittal, pathmnist, pneumoniamnist, retinamnist')
                    dataset_name = input('Specify subset of medmnsit dataset! (Type help for information)')
            except ValueError: 
                print('Give one of the listed subsets!')
                continue
            else:
                break
        
        
        info = INFO[dataset_name]
        task = info['task']
        n_channels = info['n_channels']
        n_classes = len(info['label'])
        val_auc_list = []
        epoch_old = 0
        auc_old = 0
        flag = dataset_name
        dataset_name = dataset_name + "_" + str(train_size)

        dir_path = os.path.join(output_root, '%s_checkpoints' % (dataset_name))
        print(dir_path)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        
        
        if model_input == 'resnet18':
            model = ResNet18(in_channels=n_channels, num_classes=n_classes).to(device)
            train_dataset, train_loader, val_dataset, val_loader, test_dataset, test_loader = prepareMedmnist(flag, data_root, output_root, 28, augmentations, batch_size, train_size, download)
            print('using ResNet18')
        elif model_input == 'resnet50':
            model = ResNet50(in_channels=n_channels, num_classes=n_classes).to(device)
            print('using ResNet50')
        elif model_input == 'efficientnet':
            train_dataset, train_loader, val_dataset, val_loader, test_dataset, test_loader = prepareMedmnist(flag, data_root, output_root, 224, augmentations, batch_size, train_size, download)

            model = EfficientNet.from_name('efficientnet-b0',n_channels)
            model._fc= nn.Linear(1280, n_classes)
            model.to(device)
            print('using EfficientNet-b0')

        if task == "multi-label, binary-class":
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()

        if optimizer == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
            print('optimizer is SGD')
        elif optimizer == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
            print('optimizer is Adam')
        elif optimizer == 'RMSprop':
            optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum, centered=False)
            print('optimizer is RMSprop')
        
        for epoch in trange(0, num_epoch):
            train(model, optimizer, criterion, train_loader, device, task)
            epoch_return, auc_return = val(model, val_loader, device, val_auc_list, task, dir_path, epoch, auc_old, epoch_old)
            if auc_return > auc_old:
                epoch_old = epoch_return
                auc_old = auc_return

        auc_list = np.array(val_auc_list)
        index = auc_list.argmax()
        print('epoch %s is the best model' % (index))

        print('==> Testing model...')
        restore_model_path = os.path.join(
            dir_path, 'ckpt_%d_auc_%.5f.pth' % (index, auc_list[index]))

        model.load_state_dict(torch.load(restore_model_path)['net'])
        test(model,
            'train',
            train_loader,
            device,
            dataset_name,
            task,
            output_root=output_root)
        test(model, 'val', val_loader, device, dataset_name, task, output_root=output_root)
        test(model,
            'test',
            test_loader,
            device,
            dataset_name,
            task,
            output_root=output_root)
        
        save_best_model_path = os.path.join( 
            os.path.join(output_root, dataset_name), 'ckpt_%d_auc_%.5f.pth' % (index, auc_list[index]))
        copyfile(restore_model_path, save_best_model_path)
        shutil.rmtree(dir_path)



    elif data_name == 'cifar10':
        print('cifar10')
    else:
        print('nothing')

         



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Definition of training')
    parser.add_argument('--dataset_name',
                        default='medmnist',
                        help='name of the used dataset',
                        type=str)
    parser.add_argument('--data_root',
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
    parser.add_argument('--batch_size',
                        default=128,
                        help='size of batches',
                        type=int)
    parser.add_argument('--learning_rate',
                        default=0.01,
                        help='learning rate of the optimizer',
                        type=float)
    parser.add_argument('--momentum',
                        default=0.9,
                        help='momentum of optimizer',
                        type=float)
    parser.add_argument('--train_size',
                        default=1.0,
                        help='size of trainingdata',
                        type=float)
    parser.add_argument('--weight_decay',
                        default=0,
                        help='adds L2 penalty to the cost which leads to smaller model weights',
                        type=float)
    parser.add_argument('--model_input',
                        default='resnet18',
                        help='used model',
                        type=str)
    parser.add_argument('--optimizer',
                        default='SQD',
                        help='used optimizer',
                        type=str)
    parser.add_argument('--loss_function',
                        default='CrossEntropyLoss',
                        help='used loss function',
                        type=str)                  
    parser.add_argument('--augmentations',
                        help='list of possible augmentations [crop, flip, colorjitter]',
                        nargs='*')
    parser.add_argument('--download',
                        default=True,
                        help='whether download the dataset or not',
                        type=bool)

    args = parser.parse_args()
    dataset_name = args.dataset_name.lower()
    data_root = args.data_root
    output_root = args.output_root
    num_epoch = args.num_epoch
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    momentum = args.momentum
    train_size = args.train_size
    weight_decay = args.weight_decay
    model_input = args.model_input
    optimizer = args.optimizer
    loss_function = args.loss_function
    augmentations = args.augmentations
    download = args.download

    main(dataset_name,
         data_root,
         output_root,
         num_epoch = num_epoch,
         batch_size = batch_size,
         learning_rate = learning_rate,
         momentum = momentum,
         train_size = train_size,
         weight_decay = weight_decay,
         model_input = model_input,
         optimizer = optimizer,
         loss_function = loss_function,
         augmentations = augmentations,
         download = download)
        

        