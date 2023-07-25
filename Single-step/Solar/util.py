"""
This script provides utilities for loading data and optimizing the LightCTS model. It includes the following main components:

1. `normal_std` function: Computes the corrected sample standard deviation of an array.

2. `DataLoaderS` class: Helps in loading, normalizing, and splitting the data into training, validation, and test sets. It also batches the data for training and evaluation.

3. `Optim` class: Defines an optimizer with learning rate scheduling. It supports various optimization methods, including SGD, Adagrad, Adadelta, and Adam.

"""

import numpy as np
import torch
from torch.autograd import Variable
import torch.optim as optim
import math

def normal_std(x):
    # Compute the corrected sample standard deviation of an array
    return x.std() * np.sqrt((len(x) - 1.)/(len(x)))

# DataLoader for loading, normalizing, and splitting data
class DataLoaderS(object):
    def __init__(self, file_name, train, valid, device, horizon, window):
        self.P = window
        self.h = horizon
        fin = open(file_name)
        self.rawdat = np.loadtxt(fin, delimiter=',')
        self.dat = np.zeros(self.rawdat.shape)
        self.n, self.m = self.dat.shape
        self.normalize = 2
        self.scale = np.ones(self.m)
        self._normalized()
        self._split(int(train * self.n), int((train + valid) * self.n), self.n)

        self.scale = torch.from_numpy(self.scale).float()
        tmp = self.test[1] * self.scale.expand(self.test[1].size(0), self.m)

        self.scale = self.scale.to(device)
        self.scale = Variable(self.scale)

        self.rse = normal_std(tmp)
        self.rae = torch.mean(torch.abs(tmp - torch.mean(tmp)))
        self.rmse = torch.sum(torch.abs(tmp))

        self.device = device

    # normalized by the maximum value of entire matrix.
    def _normalized(self):

        for i in range(self.m):
            self.scale[i] = np.max(np.abs(self.rawdat[:, i]))
            self.dat[:, i] = self.rawdat[:, i] / np.max(np.abs(self.rawdat[:, i]))


    # Split the data into training, validation, and test sets
    def _split(self, train, valid, test):

        train_set = range(self.P + self.h - 1, train)
        valid_set = range(train, valid)
        test_set = range(valid, self.n)
        self.train = self._batchify(train_set, self.h)
        self.valid = self._batchify(valid_set, self.h)
        self.test = self._batchify(test_set, self.h)

    # Batchify the data
    def _batchify(self, idx_set, horizon):
        n = len(idx_set)
        X = torch.zeros((n, self.P, self.m))
        Y = torch.zeros((n, self.m))
        for i in range(n):
            end = idx_set[i] - self.h + 1
            start = end - self.P
            X[i, :, :] = torch.from_numpy(self.dat[start:end, :])
            Y[i, :] = torch.from_numpy(self.dat[idx_set[i], :])
        return [X, Y]

    # Get batches of data
    def get_batches(self, inputs, targets, batch_size, shuffle=True):
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt]
            Y = targets[excerpt]
            X = X.to(self.device)
            Y = Y.to(self.device)
            yield Variable(X), Variable(Y)
            start_idx += batch_size


# Optimizer with learning rate scheduling
class Optim(object):
    # Make optimizer
    def _makeOptimizer(self):
        self.optimizer = optim.Adam(self.params, lr=self.lr, weight_decay=self.lr_decay)

    # Initialize the optimizer
    def __init__(self, params, lr, clip, lr_decay=1, start_decay_at=None):
        self.params = params
        self.last_ppl = None
        self.lr = lr
        self.clip = clip
        self.lr_decay = lr_decay
        self.start_decay_at = start_decay_at
        self.start_decay = False

        self._makeOptimizer()

    # Perform a single optimization step
    def step(self):
        # Compute gradients norm.
        grad_norm = 0
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.params, self.clip)

        self.optimizer.step()
        return  grad_norm
