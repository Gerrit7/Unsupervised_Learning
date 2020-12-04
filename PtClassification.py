import os
from PIL import Image

class PtClassification():
    def __init__(self):
        placeholder = 0

    def readPrep(self, dataDir='../resized'):
        # read Medical MNIST Data and prepare data-set
        classNames = os.listdir(dataDir)  # Each type of image can be found in its own subdirectory
        numClass = len(classNames)        # Number of types = number of subdirectories
        imageFiles = [[os.path.join(dataDir,classNames[i],x) for x in os.listdir(os.path.join(dataDir,classNames[i]))]
                    for i in range(numClass)]                     # A nested list of filenames
        numEach = [len(imageFiles[i]) for i in range(numClass)]     # A count of each type of image
        imageFilesList = []               # Created an un-nested list of filenames
        imageClass = []                   # The labels -- the type of each individual image in the list
        for i in range(numClass):
            imageFilesList.extend(imageFiles[i])
            imageClass.extend([i]*numEach[i])
        numTotal = len(imageClass)        # Total number of images
        imageWidth, imageHeight = Image.open(imageFilesList[0]).size         # The dimensions of each image

        return (imageFilesList, imageClass, imageWidth, imageHeight)
        print("There are",numTotal,"images in",numClass,"distinct categories")
        print("Label names:",classNames)
        print("Label counts:",numEach)
        print("Image dimensions:",imageWidth,"x",imageHeight)