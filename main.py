# coding: utf-8

import numpy as np
import pandas as pd

import torch as t
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms


from torch.autograd import Variable

IMG_SIZE = 400
HIDDEN_SIZE = 128
CLASS_SIZE = 90

EPOCH = 15
BATCH_SIZE = 32

LR = 0.005
MOMENTUM = 0.9

class DataSet(data.Dataset):
    def __init__(self, csv_file):
        super(DataSet, self).__init__()

        raw_data = pd.read_csv(csv_file)

        self.image = raw_data.iloc[:,12:].values.astype(np.float32)
        self.target = t.from_numpy(raw_data.iloc[:,2].values)

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        return self.image[index], self.target[index]

class FullConnected(nn.Module):
    def __init__(self, train_loader, test_loader):
        super(FullConnected, self).__init__()

        self.train_loader = train_loader
        self.test_loader = test_loader

        self.visible = nn.Sequential(
            nn.Linear(IMG_SIZE, HIDDEN_SIZE),
            nn.BatchNorm1d(HIDDEN_SIZE),
            nn.ReLU()
        )

        self.hidden = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, CLASS_SIZE),
            nn.ReLU()
        )

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=LR, momentum=MOMENTUM)

    def forward(self, x):
        x = self.visible(x)
        x = self.hidden(x)
        return x

    def solve(self):
        for epoch in range(EPOCH):
            print ('IN EPOCH: %3d' % (epoch + 1))

            for i, (image, target) in enumerate(self.train_loader):
                image, target = Variable(image), Variable(target)
                self.optimizer.zero_grad()

                loss = self.criterion(self(image), target)

                loss.backward()
                self.optimizer.step()

                print('\tBATCH: %5d\t#Loss: %.3f' % (i + 1, 1.0 * loss.data.item() / BATCH_SIZE))

            self.evaluate(epoch)
    
    def evaluate(self, epoch):
        self.eval()
        
        loss = 0.0
        correct = 0.0
        for data, target in self.test_loader:
            data, target = Variable(data), Variable(target)

            output = self(data)
            loss += F.cross_entropy(output, target).data.item()
            # prediction = output.data.max(1, keepdim=True)
            prediction = t.max(output.data, 1)[1]
            correct += prediction.eq(target.data.view_as(prediction)).cpu().sum()

        loss /= len(self.test_loader.dataset)

        print('Epoch: %3d\tAccuracy: %.3f%%' % (epoch+1, 100.0 * correct.item() / len(self.test_loader.dataset)))

if __name__ == '__main__':
    train_set = DataSet('./data/train.csv')
    # train_set = DataSet('./data/test.csv')
    test_set = DataSet('./data/test.csv')

    train_loader = data.DataLoader(
        dataset = train_set,
        batch_size = BATCH_SIZE,
        shuffle = True,
        num_workers = 2
    )

    test_loader = data.DataLoader(
        dataset = test_set,
        batch_size = BATCH_SIZE,
        shuffle = False
    )

    fc = FullConnected(train_loader, test_loader)
    fc.solve()