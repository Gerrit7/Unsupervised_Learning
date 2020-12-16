import os

slices = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
os.system('python train.py '+
'--data_name retinamnist ' +
'--input_root ../../../DataSets/medmnist-20201211T084149Z-001/medmnist/ ' +
'--output_root ../../../TrainedNets/Training_Resnet18_self ' +
'--num_epoch 100 ' +
'--train_size 0.5 ' +
'--download False '
)