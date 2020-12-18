import os
import argparse
from tqdm import trange
import numpy as np
import shutil
from shutil import copyfile
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms

from medmnist.models import ResNet18, ResNet50
from medmnist.dataset import PathMNIST, ChestMNIST, DermaMNIST, OCTMNIST, PneumoniaMNIST, RetinaMNIST, BreastMNIST, OrganMNISTAxial, OrganMNISTCoronal, OrganMNISTSagittal
from medmnist.evaluator import getAUC, getACC, save_results
from medmnist.info import INFO


def main(flag, input_model):
    ''' main function
    :param flag: name of subset

    '''

    flag_to_class = {
        "pathmnist": PathMNIST,
        "chestmnist": ChestMNIST,
        "dermamnist": DermaMNIST,
        "octmnist": OCTMNIST,
        "pneumoniamnist": PneumoniaMNIST,
        "retinamnist": RetinaMNIST,
        "breastmnist": BreastMNIST,
        "organmnist_axial": OrganMNISTAxial,
        "organmnist_coronal": OrganMNISTCoronal,
        "organmnist_sagittal": OrganMNISTSagittal,
    }
    DataClass = flag_to_class[flag]

    info = INFO[flag]
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])

    start_epoch = 0
    lr = 0.001
    batch_size = 8 #128
    val_auc_list = []
    train_dataset_scaled = []
    epoch_old = 0
    auc_old = 0

    if trainSize != 1:
        flag = flag + "_" + str(trainSize)

    dir_path = os.path.join(output_root, '%s_checkpoints' % (flag))
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    print('==> Preparing data...')

    test_transform = transforms.Compose(
        [transforms.Resize(256),
        transforms.CenterCrop(224),transforms.ToTensor(),
         transforms.Normalize(mean=[.5], std=[.5])])

    test_dataset = DataClass(root=input_root,
                                   split='test',
                                   transform=test_transform,
                                   download=download)
    test_loader = data.DataLoader(dataset=test_dataset,
                                  batch_size=batch_size,
                                  shuffle=True)
    print('Test: ', len(test_dataset))
    
    
    print('==> Testing model...')
    restore_model_path = os.path.join(
        dir_path, 'ckpt_%d_auc_%.5f.pth' % (index, auc_list[index]))

    model.load_state_dict(torch.load(restore_model_path)['net'])
    test(model,
            'train',
            train_loader,
            device,
            flag,
            task,
            output_root=output_root)
    test(model, 'val', val_loader, device, flag, task, output_root=output_root)
    test(model,
            'test',
            test_loader,
            device,
            flag,
            task,
            output_root=output_root)

    save_best_model_path = os.path.join( 
        os.path.join(output_root, flag), 'ckpt_%d_auc_%.5f.pth' % (index, auc_list[index]))
    copyfile(restore_model_path, save_best_model_path)
    shutil.rmtree(dir_path)

def test(model, split, data_loader, device, flag, task, output_root=None):
    ''' testing function
    :param model: the model to test
    :param split: the data to test, 'train/val/test'
    :param data_loader: DataLoader of data
    :param device: cpu or cuda
    :param flag: subset name
    :param task: task of current dataset, binary-class/multi-class/multi-label, binary-class

    '''

    model.eval()
    y_true = torch.tensor([]).to(device)
    y_score = torch.tensor([]).to(device)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            outputs = model(inputs.to(device))

            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32).to(device)
                m = nn.Sigmoid()
                outputs = m(outputs).to(device)
            else:
                targets = targets.squeeze().long().to(device)
                m = nn.Softmax(dim=1)
                outputs = m(outputs).to(device)
                targets = targets.float().resize_(len(targets), 1)

            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, outputs), 0)

        y_true = y_true.cpu().numpy()
        y_score = y_score.detach().cpu().numpy()
        auc = getAUC(y_true, y_score, task)
        acc = getACC(y_true, y_score, task)
        print('%s AUC: %.5f ACC: %.5f' % (split, auc, acc))

        if output_root is not None:
            output_dir = os.path.join(output_root, flag)
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            output_path = os.path.join(output_dir, '%s.csv' % (split))
            save_results(y_true, y_score, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='RUN Baseline model of MedMNIST')
    parser.add_argument('--data_name',
                        default='pathmnist',
                        help='subset of MedMNIST',
                        type=str)
    parser.add_argument('--input_root',
                        default='./input',
                        help='input root, the source of dataset files',
                        type=str)
    parser.add_argument('--output_root',
                        default='./output',
                        help='output root, where to save models and results',
                        type=str)
    parser.add_argument('--num_epoch',
                        default=100,
                        help='num of epochs of training',
                        type=int)
    parser.add_argument('--train_size',
                        default=1.0,
                        help='size of trainingdata',
                        type=float)
    parser.add_argument('--download',
                        default=True,
                        help='whether download the dataset or not',
                        type=bool)

    args = parser.parse_args()
    data_name = args.data_name.lower()
    input_root = args.input_root
    output_root = args.output_root
    end_epoch = args.num_epoch
    train_size = args.train_size
    download = args.download
    main(data_name,
         input_root,
         output_root,
         end_epoch=end_epoch,
         trainSize = train_size,
         download=download)
