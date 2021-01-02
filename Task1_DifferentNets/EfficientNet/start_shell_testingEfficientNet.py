import os

#slices = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
slices = [0.1, 0.25, 0.5, 0.75, 1.0]
dataSets    = ["pathmnist", "retinamnist", "dermamnist"]
#["octmnist", "pneumoniamnist", "pathmnist", "chestmnist","retinamnist", "dermamnist", 
#"breastmnist" ,"organmnist_coronal", "organmnist_sagittal", "organmnist_axial"]

for dset in range(len(dataSets)):
    for slc in slices:
        os.system('python testingEfficientNet.py '+
        '--data_name ' + str(dataSets[dset]) + ' ' +
        '--dataset_root ../../../DataSets/medmnist/ ' +
        '--model_root ../../../TrainedNets/EfficientNet/ ' +
        '--train_size ' + str(slc) + ' ' +
        '--download False '
        )