"""
This 'trainer' script serves as the training engine for the lightCTS model. It includes two main parts:

1. The 'train' method sets the model to training mode, performs forward pass to get model's predictions, calculates loss and performs model optimization.

2. The 'eval' method sets the model to evaluation mode, performs forward pass to get model's predictions, and calculates and returns the evaluation metrics.

This script is essential for training and evaluating the lightCTS model on data. It is used to run the training iterations, update the model parameters based on computed loss, and evaluate the performance of the model.
"""

# Import necessary libraries
import torch.optim as optim
from LightCTS_model import *
import util
import torch

class trainer():
    def __init__(self, scaler, in_dim, seq_length, nhid, nglf, lrate, wdecay, device, supports, group):
        # Initialize the model with given parameters
        self.model = LightCTS(supports=supports, in_dim=in_dim, out_dim=seq_length, hid_dim=nhid, nglf=nglf, group=group)
        self.model.to(device)
        # Set the optimizer as Adam, with given learning rate and weight decay
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        # Use masked mean absolute error as the loss function
        self.loss = util.masked_mae
        self.scaler = scaler
        self.clip = 3


    # The training function
    def train(self, input, real_val):
        self.model.train()  # Set the model to training mode
        self.optimizer.zero_grad()  # Clear the gradients of all optimized variables
        # Forward pass: get the model's predictions
        output = self.model(input)
        output = output.transpose(1, 3)
        real = torch.unsqueeze(real_val,dim=1)
        # Inverse transform the output using the scaler
        predict = self.scaler.inverse_transform(output)
        # Calculate the loss
        loss = self.loss(predict, real, 0.0)
        # Backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # Perform a single optimization step (parameter update)
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        # Calculate the metrics for the predictions
        mape = util.masked_mape(predict, real, 0.0).item()
        mae = util.masked_mae(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        # Return the metrics
        return mae, mape, rmse

    # The evaluation function
    def eval(self, input, real_val):
        self.model.eval()  # Set the model to evaluation mode
        # Forward pass: get the model's predictions
        output = self.model(input)
        output = output.transpose(1, 3)
        real = torch.unsqueeze(real_val, dim=1)
        # Inverse transform the output using the scaler
        predict = self.scaler.inverse_transform(output)
        # Calculate the metrics for the predictions
        mape = util.masked_mape(predict, real, 0.0).item()
        mae = util.masked_mae(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        # Return the metrics
        return mae, mape, rmse
