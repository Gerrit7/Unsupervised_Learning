import torch
import torch.utils.data as data
import torchvision.transforms as transforms    
import os

from datasets.medmnist.dataset import PathMNIST, ChestMNIST, DermaMNIST, OCTMNIST, PneumoniaMNIST, RetinaMNIST, \
    BreastMNIST, OrganMNISTAxial, OrganMNISTCoronal, OrganMNISTSagittal
from datasets.medmnist.info import INFO

def prepareMedmnist(flag, input_root, output_root, net_input, image_size, augmentations, batch_size, trainSize, download):
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
    transforms.CenterCrop
    flag = flag + "_" + str(trainSize)

    dir_path = os.path.join(output_root, '%s_checkpoints' % (flag))
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


    aug_values = {
	"CenterCrop"   : {"size": image_size},
	"ColorJitter"  : {"brightness": 0, "contrast": 0, "saturation": 0, "hue": 0},
	"GaussianBlur" : {"kernel": [3,3], "sigma" : 0.1},
	"Normalize"    : {"mean": [0.5], "std": [0.5]},
	"RandomHorizontalFlip" : {"probability": 0.5},
	"RandomVerticalFlip" : {"probability": 0.5},
	"RandomRotation" : {"degrees": [-20, 20]}	
    }

    tranform_compose_list = [transforms.ToTensor()]
    if net_input == "EfficientNet-b0" or net_input == "EfficientNet-b1" or net_input == "EfficientNet-b7":
        tranform_compose_list.append(transforms.Resize(256))
    for aug in augmentations:
        if aug == "centerCrop":
            tranform_compose_list.append(transforms.CenterCrop(
                        aug_values["CenterCrop"]["size"]))
        elif aug == "colorJitter":
            tranform_compose_list.append(transforms.ColorJitter(
                        brightness=aug_values["ColorJitter"]["brightness"], 
                        contrast=aug_values["ColorJitter"]["contrast"],
                        saturation=aug_values["ColorJitter"]["saturation"], 
                        hue=aug_values["ColorJitter"]["hue"]))
        elif aug == "gaussianBlur":
            tranform_compose_list.append(transforms.GaussianBlur(
                        kernel_size=aug_values["GaussianBlur"]["kernel"], 
                        sigma=aug_values["GaussianBlur"]["sigma"]))
        elif aug =="normalize":
            tranform_compose_list.append(transforms.Normalize(
                        mean=aug_values["Normalize"]["mean"], 
                        std=aug_values["Normalize"]["std"]))
        elif aug =="randomHorizontalFlip":
            tranform_compose_list.append(transforms.RandomHorizontalFlip(
                        p=aug_values["RandomHorizontalFlip"]["probability"]))
        elif aug =="randomVerticalFlip":
            tranform_compose_list.append(transforms.RandomVerticalFlip(
                        p=aug_values["RandomVerticalFlip"]["probability"]))
        elif aug =="randomRotation":
            tranform_compose_list.append(transforms.RandomRotation(
                        degrees=aug_values["RandomRotation"]["degrees"]))
        else:
            print("augmentation not found!")
    
    print('==> Preparing data...')
    
    train_transform = transforms.Compose(tranform_compose_list)
    print(train_transform)
    print(type(train_transform))
    val_transform = transforms.Compose(tranform_compose_list)

    test_transform = transforms.Compose(tranform_compose_list)

    
    train_dataset = DataClass(root=input_root,
                                    split='train',
                                    transform=train_transform,
                                    download=download)

    indices = torch.randperm(len(train_dataset))[:round(len(train_dataset)*trainSize)]
    for idx in indices:
        train_dataset_scaled.append(train_dataset[idx])

    train_loader = data.DataLoader(dataset=train_dataset_scaled,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   drop_last=True)
    val_dataset = DataClass(root=input_root,
                                  split='val',
                                  transform=val_transform,
                                  download=download)
    val_loader = data.DataLoader(dataset=val_dataset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 drop_last=True)
    test_dataset = DataClass(root=input_root,
                                   split='test',
                                   transform=test_transform,
                                   download=download)
    test_loader = data.DataLoader(dataset=test_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  drop_last=True)
    print('Train: ', len(train_dataset_scaled), ', Valid: ', len(val_dataset), ', Test: ', len(test_dataset))
    
    return train_dataset, train_loader, val_dataset, val_loader, test_dataset, test_loader
    

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(image_size),
        transforms.CenterCrop(aug_values["CenterCrop"]["size"]),
        transforms.ColorJitter(brightness=aug_values["ColorJitter"]["brightness"], contrast=aug_values["ColorJitter"]["contrast"],
                            saturation=aug_values["ColorJitter"]["saturation"], hue=aug_values["ColorJitter"]["hue"]),
        transforms.GaussianBlur(kernel_size=aug_values["GaussianBlur"]["kernel"], sigma=aug_values["GaussianBlur"]["sigma"]),
        transforms.Normalize(mean=aug_values["Normalize"]["mean"], std=aug_values["Normalize"]["std"]),
        transforms.RandomHorizontalFlip(p=aug_values["RandomHorizontalFlip"]["probability"]),
        transforms.RandomVerticalFlip(p=aug_values["RandomVerticalFlip"]["probability"]),
        transforms.RandomRotation(degrees=aug_values["RandomRotation"]["degrees"])
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(image_size),
        transforms.CenterCrop(aug_values["CenterCrop"]["size"]),
        transforms.ColorJitter(brightness=aug_values["ColorJitter"]["brightness"], contrast=aug_values["ColorJitter"]["contrast"],
                            saturation=aug_values["ColorJitter"]["saturation"], hue=aug_values["ColorJitter"]["hue"]),
        transforms.GaussianBlur(kernel_size=aug_values["GaussianBlur"]["kernel"], sigma=aug_values["GaussianBlur"]["sigma"]),
        transforms.Normalize(mean=aug_values["Normalize"]["mean"], std=aug_values["Normalize"]["std"]),
        transforms.RandomHorizontalFlip(p=aug_values["RandomHorizontalFlip"]["probability"]),
        transforms.RandomVerticalFlip(p=aug_values["RandomVerticalFlip"]["probability"]),
        transforms.RandomRotation(degrees=aug_values["RandomRotation"]["degrees"])
    ])

    
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
    