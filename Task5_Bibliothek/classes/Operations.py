import torch 
import torch.nn as nn
import os
from tqdm import trange


from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

class SupervisedLearning():
    def __init__(self, model, optimizer, scheduler, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler

    
    def train_labeled(self, train_loader, task):

        self.model.train()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            print(batch_idx)
            self.optimizer.zero_grad()
            outputs = self.model(inputs.to(self.device))

            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32).to(self.device)
                loss = self.criterion(outputs, targets)
            else:
                targets = targets.squeeze().long().to(self.device)
                loss = self.criterion(outputs, targets)

            loss.backward()
            self.optimizer.step()
        
        return self.optimizer, loss


    def val_labeled(self, val_loader, val_auc_list, task, dir_path, epoch, auc_old, epoch_old, loss):

        self.model.eval()
        y_true = torch.tensor([]).to(self.device)
        y_score = torch.tensor([]).to(self.device)
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(val_loader):
                outputs = self.model(inputs.to(self.device))

                if task == 'multi-label, binary-class':
                    targets = targets.to(torch.float32).to(self.device)
                    m = nn.Sigmoid()
                    outputs = m(outputs).to(self.device)
                else:
                    targets = targets.squeeze().long().to(self.device)
                    m = nn.Softmax(dim=1)
                    outputs = m(outputs).to(self.device)
                    targets = targets.float().resize_(len(targets), 1)

                y_true = torch.cat((y_true, targets), 0)
                y_score = torch.cat((y_score, outputs), 0)

            y_true = y_true.cpu().numpy()
            y_score = y_score.detach().cpu().numpy()
            auc = self.getAUC(y_true, y_score, task)
            val_auc_list.append(auc)

        state = {
            'net': self.model.state_dict(),
            'auc': auc,
            'epoch': epoch,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'loss': loss
        }
        
        if auc > auc_old:
            path = os.path.join(dir_path, 'ckpt_%d_auc_%.5f.pth' % (epoch, auc))
            torch.save(state, path)
            if epoch>0:
                del_path = os.path.join(dir_path, 'ckpt_%d_auc_%.5f.pth' % (epoch_old, auc_old))
                os.remove(del_path) 

        return (epoch, auc)


    def test_labeled(self, split, data_loader, flag, task, output_root=None):
        ''' testing function
        :param model: the model to test
        :param split: the data to test, 'train/val/test'
        :param data_loader: DataLoader of data
        :param device: cpu or cuda
        :param flag: subset name
        :param task: task of current dataset, binary-class/multi-class/multi-label, binary-class

        '''

        self.model.eval()
        y_true = torch.tensor([]).to(self.device)
        y_score = torch.tensor([]).to(self.device)

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(data_loader):
                outputs = self.model(inputs.to(self.device))

                if task == 'multi-label, binary-class':
                    targets = targets.to(torch.float32).to(self.device)
                    m = nn.Sigmoid()
                    outputs = m(outputs).to(self.device)
                else:
                    targets = targets.squeeze().long().to(self.device)
                    m = nn.Softmax(dim=1)
                    outputs = m(outputs).to(self.device)
                    targets = targets.float().resize_(len(targets), 1)

                y_true = torch.cat((y_true, targets), 0)
                y_score = torch.cat((y_score, outputs), 0)

            y_true = y_true.cpu().numpy()
            y_score = y_score.detach().cpu().numpy()
            auc = self.getAUC(y_true, y_score, task)
            acc = self.getACC(y_true, y_score, task)
            print('%s AUC: %.5f ACC: %.5f' % (split, auc, acc))

            if output_root is not None:
                output_dir = os.path.join(output_root, flag)
                if not os.path.exists(output_dir):
                    os.mkdir(output_dir)
                output_path = os.path.join(output_dir, '%s.csv' % (split))
                self.save_results(y_true, y_score, output_path)



    def getAUC(self, y_true, y_score, task):
        '''AUC metric.
        :param y_true: the ground truth labels, shape: (n_samples, n_classes) for multi-label, and (n_samples,) for other tasks
        :param y_score: the predicted score of each class, shape: (n_samples, n_classes)
        :param task: the task of current dataset

        '''
        if task == 'binary-class':
            y_score = y_score[:,-1]
            return roc_auc_score(y_true, y_score)
        elif task == 'multi-label, binary-class':
            auc = 0
            for i in range(y_score.shape[1]):
                label_auc = roc_auc_score(y_true[:, i], y_score[:, i])
                auc += label_auc
            return auc / y_score.shape[1]
        else:
            auc = 0
            zero = np.zeros_like(y_true)
            one = np.ones_like(y_true)
            for i in range(y_score.shape[1]):
                y_true_binary = np.where(y_true == i, one, zero)
                y_score_binary = y_score[:, i]
                auc += roc_auc_score(y_true_binary, y_score_binary)
            return auc / y_score.shape[1]


    def getACC(self, y_true, y_score, task, threshold=0.5):
        '''Accuracy metric.
        :param y_true: the ground truth labels, shape: (n_samples, n_classes) for multi-label, and (n_samples,) for other tasks
        :param y_score: the predicted score of each class, shape: (n_samples, n_classes)
        :param task: the task of current dataset
        :param threshold: the threshold for multilabel and binary-class tasks

        '''
        if task == 'multi-label, binary-class':
            zero = np.zeros_like(y_score)
            one = np.ones_like(y_score)
            y_pre = np.where(y_score < threshold, zero, one)
            acc = 0
            for label in range(y_true.shape[1]):
                label_acc = accuracy_score(y_true[:, label], y_pre[:, label])
                acc += label_acc
            return acc / y_true.shape[1]
        elif task == 'binary-class':
            y_pre = np.zeros_like(y_true)
            for i in range(y_score.shape[0]):
                y_pre[i] = (y_score[i][-1] > threshold)
            return accuracy_score(y_true, y_pre)
        else:
            y_pre = np.zeros_like(y_true)
            for i in range(y_score.shape[0]):
                y_pre[i] = np.argmax(y_score[i])
            return accuracy_score(y_true, y_pre)

    
    
    def save_results(self, y_true, y_score, outputpath):
        '''Save ground truth and scores
        :param y_true: the ground truth labels, shape: (n_samples, n_classes) for multi-label, and (n_samples,) for other tasks
        :param y_score: the predicted score of each class, shape: (n_samples, n_classes)
        :param outputpath: path to save the result csv

        '''
        idx = []

        idx.append('id')

        for i in range(y_true.shape[1]):
            idx.append('true_%s' % (i))
        for i in range(y_score.shape[1]):
            idx.append('score_%s' % (i))

        df = pd.DataFrame(columns=idx)
        for id in range(y_score.shape[0]):
            dic = {}
            dic['id'] = id
            for i in range(y_true.shape[1]):
                dic['true_%s' % (i)] = y_true[id][i]
            for i in range(y_score.shape[1]):
                dic['score_%s' % (i)] = y_score[id][i]

            df_insert = pd.DataFrame(dic, index = [0])
            df = df.append(df_insert, ignore_index=True)

        df.to_csv(outputpath, sep=',', index=False, header=True, encoding="utf_8_sig")