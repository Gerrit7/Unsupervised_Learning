import torch
import torch.utils.data as data
import torchvision.transforms as transforms    
import os

from datasets.medmnist.dataset import PathMNIST, ChestMNIST, DermaMNIST, OCTMNIST, PneumoniaMNIST, RetinaMNIST, \
    BreastMNIST, OrganMNISTAxial, OrganMNISTCoronal, OrganMNISTSagittal
from datasets.medmnist.info import INFO

def prepareMedmnist(flag, input_root, output_root, end_epoch, batch_size, trainSize, download):
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

    train_dataset_scaled = []

    flag = flag + "_" + str(trainSize)

    dir_path = os.path.join(output_root, '%s_checkpoints' % (flag))
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

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
    
    return train_dataset, train_loader, val_dataset, val_loader, test_dataset, test_loader
    