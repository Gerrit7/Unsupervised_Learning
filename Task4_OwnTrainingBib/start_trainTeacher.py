import os

slices = [0.1]#, 0.25, 0.5, 0.75, 1]
subsets    = ["pathmnist", "octmnist", "pneumoniamnist", "chestmnist", "retinamnist", "dermamnist", "breastmnist", "organmnist_axial", "organmnist_coronal", "organmnist_sagittal"]

augmentations = {
	"CenterCrop"   : {"size": 0},
	"ColorJitter"  : {"brightness": 0, "contrast": 0, "saturation": 0, "hue": 0},
	"GaussianBlur" : {"kernel": [3,3], "sigma" : 0.1},
	"Normalize"    : {"mean": [0.5], "std": [0.5]},
	"RandomHotizontalFlip" : {"probability": 0.5},
	"RandomVerticalFlip" : {"probability": 0.5},
	"RandomRotation" : {"degrees": [-20, 20]}	
}
augmentations = str(augmentations)
for slc in slices:        
	os.system('python trainTeacher.py '+
	'--dataset_name medmnist ' +
	'--data_root ../../DataSets/medmnist ' +
	'--output_root evaluation --num_epoch 100 ' +
	'--batch_size 128 ' +
	'--learning_rate 0.01 ' +
	'--momentum 0.9 ' +
	'--train_size ' + str(slc) + ' ' + 
	'--weight_decay 0.1 ' +
	'--model resnet18 ' +
	'--optimizer SGD ' +
	'--loss_function CrossEntropyLoss ' +
	'--augmentations GaussianBlur ' +
	'--download True'
        )
