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
    dataDir = '/Users/gerrit/Documents/Master-Thesis/resized'                                                      # path to directory of medical MNIST images
    numEpochs = 3                                                               # number of epochs
    batchSize = 300                                                             # size of batches
    t2vRatio = 1.2                                                              # Maximum allowed ratio of validation to training loss
    t2vEpochs = 3                                                               # Number of consecutive epochs before halting if validation loss exceeds above limit
    running_loss = 0.0                                                          # running loss
    lRate = 0.001                                                               # learning rate of classifier
    momentum = 0.9                                                              # adds a proportion of the previous weight changes to the current weight changes
    validFrac = 0.1                                                             # fraction of images for validation dataset
    testFrac = 0.1                                                              # fraction of images for test dataset

    numConvs1   = 5                                                             # number of channels produced by the convolution
    convSize1   = 7                                                             # size of the convolving kernel
    numConvs2   = 10                                                            # number of channels produced by the convolution
    convSize2   = 7                                                             # size of the convolving kernel

    fcSize1 = 400                                                               # size of sample
    fcSize2 = 80                                                                # size of sample

    validList = []                                                              # list for valid data
    testList = []                                                               # list for test data
    trainList = []                                                              # list for train data



    ####################        Read and Prepare Images         #################################
    classNames = os.listdir(dataDir)                                            # Each type of image can be found in its own subdirectory
    numClass = len(classNames)                                                  # Number of types = number of subdirectories
    imageFiles = [[os.path.join(dataDir,classNames[i],x) for x in os.listdir(os.path.join(dataDir,classNames[i]))]
                for i in range(numClass)]                                     # nested list of filenames
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


    # Check if imagetensor has already been created (for time saving )
    imageTensorFile = Path(os.path.dirname(os.path.realpath(__file__)) + "/imageTensor.npy")
    imageClassFile = Path(os.path.dirname(os.path.realpath(__file__)) + "/imageClass.npy")
    if imageTensorFile.is_file():                                               # if file exists load data
        with open(imageTensorFile, 'rb') as f:
            imageTensor = np.load(f)
            imageTensor = torch.tensor(imageTensor)
    else:                                                                       # if file doesn´t exist: load, scale and stack images
        imageTensor = torch.stack([scaleImage(Image.open(x)) for x in imageFilesList]) 
        with open(imageTensorFile, 'wb') as f:
            np.save(f, imageTensor)

    if imageClassFile.is_file():                                                # if file exists load data
        with open(imageClassFile, 'rb') as f:
            classTensor = np.load(f)
            classTensor = torch.tensor(classTensor)
    else:                                                                       # if file doesn´t exist: load, scale and stack images
        classTensor = torch.tensor(imageClass)                                  # Create label (Y) tensor  
        with open(imageClassFile, 'wb') as f:
            np.save(f, imageClass)


    print("Rescaled min pixel value = {:1.3}; Max = {:1.3}; Mean = {:1.3}"
        .format(imageTensor.min().item(),imageTensor.max().item(),imageTensor.mean().item()))

    for i in range(numTotal):
        rann = np.random.rand()                                                 # Randomly reassign images
        if rann < validFrac:
            validList.append(i)
        elif rann < testFrac + validFrac:
            testList.append(i)
        else:
            trainList.append(i)
            
    nTrain = len(trainList)                                                     # number of elements in training set
    nValid = len(validList)                                                     # number of elements in valid set
    nTest = len(testList)                                                       # number of elements in test set
    print("Training images =",nTrain,"Validation =",nValid,"Testing =",nTest)

    trainIds = torch.tensor(trainList)                                          # change type of list to tensor
    validIds = torch.tensor(validList)                                          # change type of list to tensor
    testIds = torch.tensor(testList)                                            # change type of list to tensor

    trainX = imageTensor[trainIds,:,:,:]                                        # slice image/label tensors in train, valid, test
    trainY = classTensor[trainIds]
    validX = imageTensor[validIds,:,:,:]
    validY = classTensor[validIds]
    testX = imageTensor[testIds,:,:,:]
    testY = classTensor[testIds]

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


    net = Net(imageWidth,imageHeight,numClass)                                  # create neural network
    criterion = nn.CrossEntropyLoss()                                           # set criterion
    optimizer = optim.SGD(net.parameters(), lr=lRate, momentum=momentum)        # set optimizer to squared gradient decent

    trainBats = nTrain // batchSize                                             # Number of training batches per epoch. Round down to simplify last batch
    validBats = nValid // batchSize                                             # Validation batches. Round down
    testBats = -(-nTest // batchSize)                                           # Testing batches. Round up to include all
    CEweights = torch.zeros(numClass)                                           # This takes into account the imbalanced dataset.
    for i in trainY.tolist():                                                   # By making rarer images count more to the loss, 
        CEweights[i].add_(1)                                                    # we prevent the model from ignoring them.
    CEweights = 1. / CEweights.clamp_(min=1.)                                   # Weights should be inversely related to count
    CEweights = (CEweights * numClass / CEweights.sum()).to(dev)                # The weights average to 1

    ####################        Training the Network            #################################
    for epoch in range(numEpochs):                                              # loop over the dataset multiple times
        net.train()
        epochLoss = 0
        
        permute = torch.randperm(nTrain)                                        # shuffle data to randomize batches
        trainX = trainX[permute,:,:,:]
        trainY = trainY[permute]
        for j in range(trainBats):                                              # iterate over batches
        
            optimizer.zero_grad()                                               # zero the parameter gradients

            batX = trainX[j*batchSize:(j+1)*batchSize,:,:,:].to(dev)            # Slice shuffled data into batches
            batY = trainY[j*batchSize:(j+1)*batchSize].to(dev)                  # .to(dev) moves these batches to the GPU
            yOut = net(batX)                                                    # Evalute predictions
            loss = F.cross_entropy(yOut, batY,weight=CEweights)                 # Compute loss
            epochLoss += loss.item()                                            # Add loss
            loss.backward()                                                     # Backpropagate loss
            optimizer.step()                                                    # Update model weights using optimizer
        
        validLoss = 0.
        permute = torch.randperm(nValid)                                        # We go through the exact same steps, without backprop / optimization
        validX = validX[permute,:,:,:]                                          # in order to evaluate the validation loss
        validY = validY[permute]
        net.eval()                                                              # Set model to evaluation mode
        with torch.no_grad():                                                   # Temporarily turn off gradient descent
            for j in range(validBats):
                optimizer.zero_grad()
                batX = validX[j*batchSize:(j+1)*batchSize,:,:,:].to(dev)
                batY = validY[j*batchSize:(j+1)*batchSize].to(dev)
                yOut = net(batX)
                validLoss += F.cross_entropy(yOut, batY,weight=CEweights).item()
        epochLoss /= trainBats                                                  # Average loss over batches and print
        validLoss /= validBats
        print("Epoch = {:-3}; Training loss = {:.4f}; Validation loss = {:.4f}".format(epoch,epochLoss,validLoss))
        if validLoss > t2vRatio * epochLoss:
            t2vEpochs -= 1                                                      # Test if validation loss exceeds halting threshold
            if t2vEpochs < 1:
                print("Validation loss too high; halting to prevent overfitting")
                break

    ####################        Testing the Network         #################################
    confuseMtx = np.zeros((numClass,numClass),dtype=int)                        # Create empty confusion matrix
    net.eval()
    with torch.no_grad():
        permute = torch.randperm(nTest)                                         # Shuffle test data
        testX = testX[permute,:,:,:]
        testY = testY[permute]
        for j in range(testBats):                                               # Iterate over test batches
            batX = testX[j*batchSize:(j+1)*batchSize,:,:,:].to(dev)
            batY = testY[j*batchSize:(j+1)*batchSize].to(dev)
            yOut = net(batX)                                                    # Pass test batch through model
            pred = yOut.max(1,keepdim=True)[1]                                  # Generate predictions by finding the max Y values
            for j in torch.cat((batY.view_as(pred), pred),dim=1).tolist():      # Glue together Actual and Predicted to
                confuseMtx[j[0],j[1]] += 1                                      # make (row, col) pairs, and increment confusion matrix
    correct = sum([confuseMtx[i,i] for i in range(numClass)])                   # Sum over diagonal elements to count correct predictions
    print("Correct predictions: ",correct,"of",nTest)
    print("Confusion Matrix:")
    print(confuseMtx)
    print(classNames)

    def scaleBack(x):                                                           # Pass a tensor, return a numpy array from 0 to 1
        if(x.min() < x.max()):                                                  # Assuming the image isn't empty, rescale so its values run from 0 to 1
            x = (x - x.min())/(x.max() - x.min())
        return x[0].to(cpu).numpy()                                             # Remove channel (grayscale anyway)


    net.eval()
    plt.subplots(3,3,figsize=(8,8))
    imagesLeft = 9
    permute = torch.randperm(nTest)                                             # Shuffle test data
    testX = testX[permute,:,:,:]
    testY = testY[permute]
    for j in range(testBats):                                                   # Iterate over test batches
        batX = testX[j*batchSize:(j+1)*batchSize,:,:,:].to(dev)
        batY = testY[j*batchSize:(j+1)*batchSize].to(dev)
        yOut = net(batX)                                                        # Pass test batch through model
        pred = yOut.max(1)[1].tolist()                                          # Generate predictions by finding the max Y values
        for i, y in enumerate(batY.tolist()):
            if imagesLeft and y != pred[i]:                                     # Compare the actual y value to the prediction
                imagesLeft -= 1
                plt.subplot(3,3,9-imagesLeft)
                plt.xlabel(classNames[pred[i]])                                 # Label image with what the model thinks it is
                plt.imshow(scaleBack(batX[i]),cmap='gray',vmin=0,vmax=1)
    plt.tight_layout()
    plt.show()

    ####################        Saving model             #################################

    torch.save(net, 'MedNIST_Tutorial_model')
