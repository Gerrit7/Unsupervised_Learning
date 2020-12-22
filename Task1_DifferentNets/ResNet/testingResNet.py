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

from medmnist.models import ResNet18, ResNet50
from medmnist.dataset import PathMNIST, ChestMNIST, DermaMNIST, OCTMNIST, PneumoniaMNIST, RetinaMNIST, BreastMNIST, OrganMNISTAxial, OrganMNISTCoronal, OrganMNISTSagittal
from medmnist.evaluator import getAUC, getACC, save_results
from medmnist.info import INFO


def main(flag, dataset_root, model_root, trainSize, download):
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
    batch_size = 8 #128
    val_auc_list = []
    train_dataset_scaled = []
    epoch_old = 0
    auc_old = 0

    print('==> Preparing data...')
    train_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[.5], std=[.5])])

    val_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[.5], std=[.5])])

    test_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[.5], std=[.5])])

    train_dataset = DataClass(root=dataset_root,
                                    split='train',
                                    transform=train_transform,
                                    download=download)

    indices = torch.randperm(len(train_dataset))[:round(len(train_dataset)*trainSize)]
    for idx in indices:
        train_dataset_scaled.append(train_dataset[idx])

    train_loader = data.DataLoader(dataset=train_dataset_scaled,
                                   batch_size=batch_size,
                                   shuffle=True)
    val_dataset = DataClass(root=dataset_root,
                                  split='val',
                                  transform=val_transform,
                                  download=download)
    val_loader = data.DataLoader(dataset=val_dataset,
                                 batch_size=batch_size,
                                 shuffle=True)
    test_dataset = DataClass(root=dataset_root,
                                   split='test',
                                   transform=test_transform,
                                   download=download)
    test_loader = data.DataLoader(dataset=test_dataset,
                                  batch_size=batch_size,
                                  shuffle=True)

    print('Train: ', len(train_dataset_scaled), ', Valid: ', len(val_dataset), ', Test: ', len(test_dataset))
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ResNet18(in_channels=n_channels, num_classes=n_classes)

    print('==> Testing model...')

    for file in os.listdir(os.path.join(model_root,'%s_%.2f' % (flag, trainSize))):
        if file.endswith(".pth"):
            restore_model_path = os.path.join(os.path.join(model_root,'%s_%.2f' % (flag, trainSize)),file)
    model.to(device)
    model.load_state_dict(torch.load(restore_model_path, map_location=device)['net'])

    test(model,
            'train',
            train_loader,
            device,
            flag,
            task,
            output_root=model_root)
    test(model, 'val', val_loader, device, flag, task, output_root=model_root)
    test(model,
            'test',
            test_loader,
            device,
            flag,
            task,
            output_root=model_root)

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

        file_exists = os.path.isfile('../../../TrainedNets/Training_Resnet18_self/results.csv')
        with open('../../../TrainedNets/Training_Resnet18_self/results.csv', 'a+', newline='') as csvfile:
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

        """ if output_root is not None:
            output_dir = os.path.join(output_root, flag)
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            output_path = os.path.join(output_dir, '%s.csv' % (split))
            save_results(y_true, y_score, output_path) """


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='RUN Baseline model of MedMNIST')
    parser.add_argument('--data_name',
                        default='pathmnist',
                        help='subset of MedMNIST',
                        type=str)
    parser.add_argument('--dataset_root',
                        default='./dataset',
                        help='input root, the source of dataset files',
                        type=str)
    parser.add_argument('--model_root',
                        default='./model_root',
                        help='model root, where the models and results are saved',
                        type=str)
    parser.add_argument('--train_size',
                        default=1.0,
                        help='size of trainingdata',
                        type=float)
    parser.add_argument('--download',
                        default=True,
                        help='whether download the dataset or not',
                        type=bool)

    args = parser.parse_args()
    data_name = args.data_name.lower()
    dataset_root = args.dataset_root
    model_root = args.model_root
    train_size = args.train_size
    download = args.download
    
    main(data_name, 
        dataset_root, 
        model_root, 
        trainSize = train_size,
        download=download)
