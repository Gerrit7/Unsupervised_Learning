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

def main(flag, input_root, output_root, end_epoch, trainSize, download, continueTrain):
    ''' main function
    :param flag: name of subset

    '''

    flag_to_class = {
        "pathmnist": PathMNIST,
        "chestmnist": ChestMNIST,
        "dermamnist": DermaMNIST,
        "octmnist": OCTMNIST,
        "pneumoniamnist": PneumoniaMNIST,
        "retinamnist": RetinaMNIST,
        "breastmnist": BreastMNIST,
        "organmnist_axial": OrganMNISTAxial,
        "organmnist_coronal": OrganMNISTCoronal,
        "organmnist_sagittal": OrganMNISTSagittal,
    }
    DataClass = flag_to_class[flag]

    info = INFO[flag]
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])

    start_epoch = 0
    lr = 0.001
    batch_size = 8
    val_auc_list = []
    train_dataset_scaled = []

    if trainSize != 1:
        flag = flag + "_" + str(trainSize)

    dir_path = os.path.join(output_root, '%s_checkpoints' % (flag))
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    print('==> Preparing data...')
    train_transform = transforms.Compose(
        [transforms.Resize(256),transforms.ToTensor(),
         transforms.Normalize(mean=[.5], std=[.5])])

    val_transform = transforms.Compose(
        [transforms.Resize(256),transforms.ToTensor(),
         transforms.Normalize(mean=[.5], std=[.5])])

    test_transform = transforms.Compose(
        [transforms.Resize(256),transforms.ToTensor(),
         transforms.Normalize(mean=[.5], std=[.5])])

    train_dataset = DataClass(root=input_root,
                                    split='train',
                                    transform=train_transform,
                                    download=download)

    indices = torch.randperm(len(train_dataset))[:round(len(train_dataset)*trainSize)]
    for idx in indices:
        train_dataset_scaled.append(train_dataset[idx])

    train_loader = data.DataLoader(dataset=train_dataset_scaled,
                                   batch_size=batch_size,
                                   shuffle=True)
    val_dataset = DataClass(root=input_root,
                                  split='val',
                                  transform=val_transform,
                                  download=download)
    val_loader = data.DataLoader(dataset=val_dataset,
                                 batch_size=batch_size,
                                 shuffle=True)
    test_dataset = DataClass(root=input_root,
                                   split='test',
                                   transform=test_transform,
                                   download=download)
    test_loader = data.DataLoader(dataset=test_dataset,
                                  batch_size=batch_size,
                                  shuffle=True)
    print('Train: ', len(train_dataset_scaled), ', Valid: ', len(val_dataset), ', Test: ', len(test_dataset))
    print('==> Building and training model...')

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
    
    model = Inception_ResNetv2(n_channels)
    model.linear= nn.Linear(1536, n_classes, True)

    if task == "multi-label, binary-class":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    if continueTrain == True:
        # loading old checkpoint for further training
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
    for epoch in trange(start_epoch, end_epoch):
        train(model, optimizer, criterion, train_loader, device, task)
        val(model, val_loader, device, val_auc_list, task, dir_path, epoch, optimizer)

    auc_list = np.array(val_auc_list)
    print(auc_list)
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
         flag,
         task,
         output_root=output_root)
    test(model, 'val', val_loader, device, flag, task, output_root=output_root)
    test(model,
         'test',
         test_loader,
         device,
         flag,
         task,
         output_root=output_root)
    print(auc_list)
    save_best_model_path = os.path.join( 
        os.path.join(output_root, flag), 'ckpt_%d_auc_%.5f.pth' % (index, auc_list[index]))
    copyfile(restore_model_path, save_best_model_path)
    shutil.rmtree(dir_path)


def train(model, optimizer, criterion, train_loader, device, task):
    ''' training function
    :param model: the model to train
    :param optimizer: optimizer used in training
    :param criterion: loss function
    :param train_loader: DataLoader of training set
    :param device: cpu or cuda
    :param task: task of current dataset, binary-class/multi-class/multi-label, binary-class

    '''

    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs.to(device))

        if task == 'multi-label, binary-class':
            targets = targets.to(torch.float32).to(device)
            loss = criterion(outputs, targets)
        else:
            targets = targets.squeeze().long().to(device)
            loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()


def val(model, val_loader, device, val_auc_list, task, dir_path, epoch, optimizer):
    ''' validation function
    :param model: the model to validate
    :param val_loader: DataLoader of validation set
    :param device: cpu or cuda
    :param val_auc_list: the list to save AUC score of each epoch
    :param task: task of current dataset, binary-class/multi-class/multi-label, binary-class
    :param dir_path: where to save model
    :param epoch: current epoch

    '''

    model.eval()
    y_true = torch.tensor([]).to(device)
    y_score = torch.tensor([]).to(device)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            outputs = model(inputs.to(device))

            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32).to(device)
                m = nn.Sigmoid()
                outputs = m(outputs).to(device)
            else:
                targets = targets.squeeze().long().to(device)
                m = nn.Softmax(dim=1)
                outputs = m(outputs).to(device)
                targets = targets.float().resize_(len(targets), 1)

            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, outputs), 0)

        y_true = y_true.cpu().numpy()
        y_score = y_score.detach().cpu().numpy()
        auc = getAUC(y_true, y_score, task)
        val_auc_list.append(auc)

    state = {
        'net': model.state_dict(),
        'auc': val_auc_list,
        'epoch': epoch + 1,
        'optimizer': optimizer.state_dict(),
    }

    path = os.path.join(dir_path, 'ckpt_%d_auc_%.5f.pth' % (epoch, auc))
    torch.save(state, path)


def test(model, split, data_loader, device, flag, task, output_root=None):
    ''' testing function
    :param model: the model to test
    :param split: the data to test, 'train/val/test'
    :param data_loader: DataLoader of data
    :param device: cpu or cuda
    :param flag: subset name
    :param task: task of current dataset, binary-class/multi-class/multi-label, binary-class

    '''

    model.eval()
    y_true = torch.tensor([]).to(device)
    y_score = torch.tensor([]).to(device)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            outputs = model(inputs.to(device))

            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32).to(device)
                m = nn.Sigmoid()
                outputs = m(outputs).to(device)
            else:
                targets = targets.squeeze().long().to(device)
                m = nn.Softmax(dim=1)
                outputs = m(outputs).to(device)
                targets = targets.float().resize_(len(targets), 1)

            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, outputs), 0)

        y_true = y_true.cpu().numpy()
        y_score = y_score.detach().cpu().numpy()
        auc = getAUC(y_true, y_score, task)
        acc = getACC(y_true, y_score, task)
        print('%s AUC: %.5f ACC: %.5f' % (split, auc, acc))

        if output_root is not None:
            output_dir = os.path.join(output_root, flag)
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            output_path = os.path.join(output_dir, '%s.csv' % (split))
            save_results(y_true, y_score, output_path)

        file_exists = os.path.isfile('../../../TrainedNets/Inception_ResNet_V2/results.csv')
        with open('../../../TrainedNets/Inception_ResNet_V2/results.csv', 'a+', newline='') as csvfile:
            fieldnames = ['Datensatz', 'Prozent des Trainingssatzes', 'Train/Validation/Test',
                            'AUC', 'ACC']
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()  # file doesn't exist yet, write a header
            writer.writerow({   'Datensatz': flag, 
                                'Prozent des Trainingssatzes': train_size,
                                'Train/Validation/Test': split,
                                'AUC' : auc,
                                'ACC' : acc})

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
