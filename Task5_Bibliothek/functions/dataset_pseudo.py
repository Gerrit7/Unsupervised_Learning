import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class PseudolabelDataset(Dataset):
    """Face Landmarks dataset."""
    
    flag = ...

    def __init__(self,
                 root,
                 transform=None,
                 target_transform=None):
        ''' dataset
        :param split: 'train', 'val' or 'test', select subset
        :param transform: data transformation
        :param target_transform: target transformation
    
        '''
        self.root = root

        print(os.path.join(self.root, "{}.npz".format(self.flag)))
        if not os.path.exists(
                os.path.join(self.root, "{}.npz".format(self.flag))):
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        npz_file = np.load(os.path.join(self.root, "{}.npz".format(self.flag)))

        self.transform = transform
        self.target_transform = target_transform

        self.img = npz_file['inputs']
        self.label = npz_file['targets']

    def __len__(self):
        return self.img.shape[0]

    def __getitem__(self, index):
        img, target = self.img[index], self.label[index].astype(int)
        img = Image.fromarray(np.uint8(img))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class PathMNIST(PseudolabelDataset):
    flag = "combined_pathmnist"


class OCTMNIST(PseudolabelDataset):
    flag = "combined_octmnist"


class PneumoniaMNIST(PseudolabelDataset):
    flag = "combined_pneumoniamnist"


class ChestMNIST(PseudolabelDataset):
    flag = "combined_chestmnist"


class DermaMNIST(PseudolabelDataset):
    flag = "combined_dermamnist"


class RetinaMNIST(PseudolabelDataset):
    flag = "combined_retinamnist"


class BreastMNIST(PseudolabelDataset):
    flag = "combined_breastmnist"


class OrganMNISTAxial(PseudolabelDataset):
    flag = "combined_organmnist_axial"


class OrganMNISTCoronal(PseudolabelDataset):
    flag = "combined_organmnist_coronal"


class OrganMNISTSagittal(PseudolabelDataset):
    flag = "combined_organmnist_sagittal"
