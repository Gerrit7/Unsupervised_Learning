import torch
import torch.utils.data as data
import torchvision.transforms as transforms    
import os
import numpy as np
import math
import random

from medmnist.dataset import PathMNIST, ChestMNIST, DermaMNIST, OCTMNIST, PneumoniaMNIST, RetinaMNIST, \
    BreastMNIST, OrganMNISTAxial, OrganMNISTCoronal, OrganMNISTSagittal
from medmnist.info import INFO


class PrepareData:
    def __init__(self, flag, input_root, output_root, net_input, download=True):
        self. flag = flag
        self.input_root = input_root
        self. output_root = output_root
        self.net_input = net_input
        self.download = download

    def createTransform(self, image_size=32, augmentations=[]):
        aug_values = {
            "CenterCrop"   : {"size": 10},
            "ColorJitter"  : {"brightness": 0, "contrast": 0, "saturation": 0, "hue": 0},
            "GaussianBlur" : {"kernel": [3,3], "sigma" : 0.1},
            "Normalize"    : {"mean": [0.5], "std": [0.5]},
            "RandomHorizontalFlip" : {"probability": 0.5},
            "RandomVerticalFlip" : {"probability": 0.5},
            "RandomRotation" : {"degrees": [-20, 20]}	
        }

        tranform_compose_list = [transforms.ToTensor()]
        if self.net_input == "EfficientNet-b0" or self.net_input == "EfficientNet-b1" or self.net_input == "EfficientNet-b7":
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
            #else:
                #print("augmentation not found!")
        
        transform = transforms.Compose(tranform_compose_list)
        return transform


    def prepareDataSet(self, split, transform, split_size):

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
        DataClass = flag_to_class[self.flag]

        dataset = DataClass(root=self.input_root,
                            split=split,
                            transform=transform,
                            download=self.download)
        
        # indices_labeled = int(math.ceil(len(dataset)*split_size))
        # indices_unlabeled = int(math.floor(len(dataset)*(1-split_size)))
        
        # # changed random_split in torch.utils.data. dataset.py for returning indices
        # [subset_labeled, indices_labeled], [subset_unlabeled, indices_unlabeled] = torch.utils.data.random_split(dataset,[indices_labeled, indices_unlabeled])

        # dataset_labeled = [dataset[i] for i in indices_labeled]
        # dataset_unlabeled = [dataset[i] for i in indices_unlabeled]
        
        # print("dataset: type: ", type(dataset), "length: ", len(dataset))
        # print("subset_labeled: type: ", type(subset_labeled), "length: ", len(subset_labeled))
        # print("dataset_labeled: type: ", type(dataset_labeled), "length: ", len(dataset_labeled))
        # print("subset_unlabeled: type: ", type(subset_unlabeled), "length: ", len(subset_unlabeled))
        # print("dataset_unlabeled: type: ", type(dataset_unlabeled), "length: ", len(dataset_unlabeled))
        
        return dataset#, subset_labeled, dataset_labeled, subset_unlabeled, dataset_unlabeled


def createDataLoader(data_in, batch_size):
    data_loader = data.DataLoader(dataset=data_in,
                                batch_size=batch_size,
                                shuffle=True)

    return data_loader

def splitDataset(dataset, data_loader, n_classes, n_train, train_size, task):
    vars()["train_counter"] = np.zeros(n_classes)
    vars()["train_class_list"] = []
    for i in range(n_classes):
        vars()["train_class_list"].append([])

    for batch_idx, (inputs, targets) in enumerate(data_loader):
            for file_idx, (value, target) in enumerate(zip(inputs, targets)):
                vars()["train_counter"][target] +=1
                vars()["train_class_list"][target].append(batch_idx+file_idx)
                if task == 'multi-label, binary-class':
                    for clas in range(n_classes):
                        if target[clas] == 1:
                            vars()["train_counter_subclass"][clas] += 1

    indices_labeled = []
    indices_unlabeled = []
    for class_list in range(n_classes):
        random_samples = random.sample(vars()["train_class_list"][class_list], int((n_train * train_size)/n_classes))
        indices_labeled = indices_labeled + random_samples
        indices_unlabeled = indices_unlabeled + list(set(vars()["train_class_list"][class_list]) - set(random_samples)) 
        
    dataset_labeled = [data_loader.dataset[i] for i in indices_labeled]
    subset_labeled = data.Subset(dataset, indices_labeled)
    dataset_unlabeled = [data_loader.dataset[i] for i in indices_unlabeled]
    subset_unlabeled = data.Subset(dataset, indices_unlabeled)
    return dataset_labeled, subset_labeled, dataset_unlabeled, subset_unlabeled

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


