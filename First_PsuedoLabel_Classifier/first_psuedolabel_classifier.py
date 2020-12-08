from __future__ import print_function
import numpy as np
import imageio
import os
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing

if __name__ == '__main__':
    ####################        Test if an GPU is available     #################################
    if torch.cuda.is_available():     
        dev = torch.device("cuda:0")
        kwar = {'num_workers': 8, 'pin_memory': True}
        cpu = torch.device("cpu")
    else:
        print("Warning: CUDA not found, CPU only.")
        dev = torch.device("cpu")
        kwar = {}
        cpu = torch.device("cpu")

    np.random.seed(551)

    ####################        Predefine some Variables        #################################
    dataDir = '../resized'                                                      # path to directory of medical MNIST images
    numEpochs = 3                                                               # number of epochs
    batchSize = 300                                                             # size of batches
    
    numConvs1   = 5                                                             # number of channels produced by the convolution
    convSize1   = 7                                                             # size of the convolving kernel
    numConvs2   = 10                                                            # number of channels produced by the convolution
    convSize2   = 7                                                             # size of the convolving kernel

    fcSize1 = 400                                                               # size of sample
    fcSize2 = 80                                                                # size of sample



    ####################        Read and Prepare Images         #################################
    classNames = os.listdir(dataDir)                                            # Each type of image can be found in its own subdirectory
    numClass = len(classNames)                                                  # Number of types = number of subdirectories
    imageFiles = [[os.path.join(dataDir,classNames[i],x) for x in os.listdir(os.path.join(dataDir,classNames[i]))]
                for i in range(numClass)]                                       # nested list of filenames
    numEach = [len(imageFiles[i]) for i in range(numClass)]                     # count of each type of image
    imageFilesList = []                                                         # un-nested list of filenames
    imageClass = []                                                             # The labels -- the type of each individual image in the list
    for i in range(numClass):
        imageFilesList.extend(imageFiles[i])
        imageClass.extend([i]*numEach[i])
    numTotal = len(imageClass)                                                  # Total number of images
    imageWidth, imageHeight = Image.open(imageFilesList[0]).size                # The dimensions of each image

    print("There are",numTotal,"images in",numClass,"distinct categories")
    print("Label names:",classNames)
    print("Label counts:",numEach)
    print("Image dimensions:",imageWidth,"x",imageHeight)

    toTensor = torchvision.transforms.ToTensor()
    def scaleImage(x):                                                          # Pass a PIL image, return a tensor
        y = toTensor(x)
        if(y.min() < y.max()):                                                  # Assuming the image isn't empty, rescale so its values run from 0 to 1
            y = (y - y.min())/(y.max() - y.min()) 
        z = y - y.mean()                                                        # Subtract the mean value of the image
        return z

    
    ####################        Define the neural network       #################################
    class Net(nn.Module):
        def __init__(self,xDim,yDim,numC):
            super(Net, self).__init__()

            self.conv1 = nn.Conv2d(1,numConvs1,convSize1)                       # first convolutional layer
            #self.pool = nn.MaxPool2d(2,2)                                      # max pooling layer
            self.conv2 = nn.Conv2d(numConvs1,numConvs2, convSize2)              # second convolutional layer

            self.fc1 = nn.Linear(numConvs2*(xDim-(convSize1-1)-(convSize2-1))*(yDim-(convSize1-1)-(convSize2-1)), fcSize1)    # first fully connected layer
            self.fc2 = nn.Linear(fcSize1,fcSize2)                               # second fully connected layer
            self.fc3 = nn.Linear(fcSize2,numClass)                              # third fully connected layer

        def forward(self, x):
            # x = self.pool(F.relu(self.conv1(x)))                              # first conv layer with relu activation function
            # x = self.pool(F.relu(self.conv2(x)))                              # second conv layer with relu activation function
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = x.view(-1, self.num_flat_features(x))
            x = F.relu(self.fc1(x))                                             # first fc layer with relu activation function
            x = F.relu(self.fc2(x))                                             # second fc layer with relu activation function
            x = self.fc3(x)                                                     # output layer
            return x
        
        def num_flat_features(self, x):                                         # Count the individual nodes in a layer
            size = x.size()[1:]
            num_features = 1
            for s in size:
                num_features *= s
            return num_features