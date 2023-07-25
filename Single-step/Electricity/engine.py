"""
This 'trainer' script serves as the training engine for the lightCTS model. It includes two main parts:

1. The 'train' method sets the model to training mode, performs forward pass to get model's predictions, calculates loss and performs model optimization.

2. The 'eval' method sets the model to evaluation mode, performs forward pass to get model's predictions, and calculates and returns the evaluation metrics.

This script is essential for training and evaluating the lightCTS model on data. It is used to run the training iterations, update the model parameters based on computed loss, and evaluate the performance of the model.
"""

# Import necessary libraries
import torch.optim as optim
from LightCTS_model import *
from util import *
import torch

class trainer():
    def __init__(self, nhid, nglf, group, lr, clip, lr_decay):
        # Initialize the model with given parameters
        self.model = LightCTS(hid_dim=nhid, nglf=nglf, group=group)
        # Set the optimizer with given learning rate and weight decay
        self.optim = Optim(self.model.parameters(), lr, clip, lr_decay)



    # Model training function with a batch of CTS data
    def train(self, data, X, Y, model, criterion, batch_size):
        model.train()
        total_loss = 0
        n_samples = 0
        for X, Y in data.get_batches(X, Y, batch_size, True):
            model.zero_grad()
            X = torch.unsqueeze(X, dim=1)
            X = X.transpose(2, 3)
            tx = X
            ty = Y
            output = model(tx)
            output = torch.squeeze(output)
            scale = data.scale.expand(output.size(0), data.m)
            loss = criterion(output * scale, ty * scale)
            loss.backward()
            total_loss += loss.item()
            n_samples += (output.size(0) * data.m)
            grad_norm = self.optim.step()

        return total_loss / n_samples

    # Model evaluation function with a batch of CTS data
    def eval(self, data, X, Y, model, evaluateL2, evaluateL1, batch_size):
        model.eval()
        total_loss = 0
        total_loss_l1 = 0
        n_samples = 0
        predict = None
        test = None

        for X, Y in data.get_batches(X, Y, batch_size, False):
            X = torch.unsqueeze(X, dim=1)
            X = X.transpose(2, 3)
            with torch.no_grad():
                output = model(X)
            output = torch.squeeze(output)
            if len(output.shape) == 1:
                output = output.unsqueeze(dim=0)
            if predict is None:
                predict = output
                test = Y
            else:
                predict = torch.cat((predict, output))
                test = torch.cat((test, Y))

            scale = data.scale.expand(output.size(0), data.m)
            total_loss += evaluateL2(output * scale, Y * scale).item()
            total_loss_l1 += evaluateL1(output * scale, Y * scale).item()
            n_samples += (output.size(0) * data.m)

        rse = math.sqrt(total_loss / n_samples) / data.rse
        rae = (total_loss_l1 / n_samples) / data.rae
        mae = total_loss_l1 / n_samples
        rmse = math.sqrt(total_loss * n_samples) / data.rmse

        predict = predict.data.cpu().numpy()
        Ytest = test.data.cpu().numpy()
        sigma_p = (predict).std(axis=0)
        sigma_g = (Ytest).std(axis=0)
        mean_p = predict.mean(axis=0)
        mean_g = Ytest.mean(axis=0)
        index = (sigma_g != 0)
        correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
        correlation = (correlation[index]).mean()

        return rse, rae, correlation, mae, rmse


