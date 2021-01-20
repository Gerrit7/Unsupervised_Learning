import torch
import csv
import numpy as np
import os

from medmnist.info import INFO
from PrepareData import PrepareData, splitDataset, createDataLoader, ConcatDataset

flags = ["octmnist", "pneumoniamnist", "pathmnist", "chestmnist","retinamnist", "dermamnist", "breastmnist", 
"organmnist_coronal", "organmnist_sagittal", "organmnist_axial"]

augmentations = []
batch_size = 128
data_root = "../../DataSets/medmnist"
output_root = "Auswertung"
net_input = "Resnet18"
train_size = 1.0




# ************************************** prepare data for training *************************************
for flag in flags:
    info = INFO[flag]
    n_channels = info['n_channels']
    n_classes = len(info['label'])
    vars()[flag + "train_counter"] = np.zeros(n_classes)
    vars()[flag + "val_counter"] = np.zeros(n_classes)
    vars()[flag + "test_counter"] = np.zeros(n_classes)

    prepareClass = PrepareData(flag, data_root, output_root, net_input, download=True)
    train_transform = prepareClass.createTransform(image_size=32, augmentations=augmentations)
    val_transform = prepareClass.createTransform(image_size=32, augmentations=augmentations)
    test_transform = prepareClass.createTransform(image_size=32, augmentations=augmentations)

    train_dataset, train_subset_labeled, train_dataset_labeled, train_subset_unlabeled, train_dataset_unlabeled = prepareClass.prepareDataSet('train', train_transform, train_size)
    train_loader_labeled   = createDataLoader(train_subset_labeled, batch_size)

    val_dataset, val_subset_labeled, val_dataset_labeled, val_subset_unlabeled, val_dataset_unlabeled = prepareClass.prepareDataSet('val', val_transform, 1)
    val_loader_labeled   = createDataLoader(val_subset_labeled, batch_size)
    #val_loader_unlabeled = createDataLoader(val_subset_unlabeled, batch_size)

    test_dataset, test_subset_labeled, test_dataset_labeled, test_subset_unlabeled, test_dataset_unlabeled = prepareClass.prepareDataSet('test', test_transform, 1)
    test_loader_labeled   = createDataLoader(test_subset_labeled, batch_size)
    #test_loader_unlabeled = createDataLoader(test_subset_unlabeled, batch_size)

    print('Train: ', len(train_subset_labeled), ', Valid: ', len(val_subset_labeled), ', Test: ', len(test_subset_labeled))


    for batch_idx, (inputs, targets) in enumerate(train_loader_labeled):
        for value, target in zip(inputs, targets):
            vars()[flag + "train_counter"][target] +=1
    print(flag, " Trainingsdaten: ", vars()[flag + "train_counter"])
    print("Summe der Trainingsdaten: ", sum(vars()[flag + "train_counter"]))

    for batch_idx, (inputs, targets) in enumerate(val_loader_labeled):
        for value, target in zip(inputs, targets):
            vars()[flag + "val_counter"][target] +=1
    print(flag, " Validierungsdaten: ", vars()[flag + "val_counter"])
    print("Summe der Validierungssdaten: ", sum(vars()[flag + "val_counter"]))

    for batch_idx, (inputs, targets) in enumerate(test_loader_labeled):
        for value, target in zip(inputs, targets):
            vars()[flag + "test_counter"][target] +=1
    print(flag, " Testdaten: ", vars()[flag + "test_counter"])
    print("Summe der Testdaten: ", sum(vars()[flag + "test_counter"]))


    splits = ["train", "val", "test"]
    # save results in csv file
    file_exists = os.path.isfile(os.path.join(output_root, 'results.csv'))
    with open(os.path.join(output_root, 'dataset_evaluation.csv'), 'a+', newline='') as csvfile:
        fieldnames = [  'dataset',
                        'split', 
                        'class', 
                        'samples'
                    ]
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()  # file doesn't exist yet, write a header
        for split in splits:
            for class_num, samples in enumerate(vars()[flag + split + "_counter"]):
                writer.writerow({   'dataset': flag, 
                                    'split': split,
                                    'class': class_num,
                                    'samples': samples,
                                })