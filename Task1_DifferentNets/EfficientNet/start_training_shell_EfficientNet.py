import os

#slices = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
slices = [0.1, 0.25, 0.5, 0.75, 1]
dataSets    = ["pathmnist", "octmnist", "pneumoniamnist"] 
dataSets2   = ["chestmnist", "retinamnist", "dermamnist", "breastmnist"]
dataSets3   = ["organmnist_axial", "organmnist_coronal", "organmnist_sagittal"]


for dset in range(len(dataSets)):
    for slc in slices:
        os.system('python trainEfficientNet.py '+
        '--data_name ' + str(dataSets[dset]) + ' ' +
        '--input_root ../../../DataSets/medmnist/ ' +
        '--output_root ../../../TrainedNets/EfficientNet ' +
        '--num_epoch 100 ' +
        '--train_size ' + str(slc) + ' ' +
        '--download False '
        )
