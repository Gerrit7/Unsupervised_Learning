import os

#slices = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
slices = [0.1, 0.25, 0.5, 0.75]
dataSets    = ["pathmnist", "octmnist", "pneumoniamnist"] 
dataSets2   = ["retinamnist", "dermamnist", "breastmnist"] #"chestmnist",
dataSets3   = ["organmnist_axial", "organmnist_coronal", "organmnist_sagittal"]


for dset in range(len(dataSets)):
    for slc in slices:
        os.system('python train.py '+
        '--data_name ' + str(dataSets[dset]) + ' ' +
        '--input_root ../../DataSets/medmnist-20201211T084149Z-001/medmnist/ ' +
        '--output_root ../../TrainedNets/Training_Resnet18_self ' +
        '--num_epoch 100 ' +
        '--train_size ' + str(slc) + ' ' +
        '--download False '
        )