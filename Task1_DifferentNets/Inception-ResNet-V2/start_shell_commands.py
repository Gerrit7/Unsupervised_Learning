import os

#slices = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
slices = [0.10, 0.25, 0.50, 0.75, 1.00]
dataSets    = ["octmnist", "pneumoniamnist", "pathmnist"]
#dataSets2   = ["chestmnist","retinamnist", "dermamnist", "breastmnist"] 
#dataSets3   = ["organmnist_coronal", "organmnist_sagittal", "organmnist_axial"]


for dset in range(len(dataSets)):
    for slc in slices:
        os.system('python train_inception_resnet_v2.py '+
        '--data_name ' + str(dataSets[dset]) + ' ' +
        '--input_root ../../../DataSets/medmnist/ ' +
        '--output_root ../../../TrainedNets/Training_Resnet18_self ' +
        '--num_epoch 100 ' +
        '--train_size ' + str(slc) + ' ' +
        '--download False '
        )
