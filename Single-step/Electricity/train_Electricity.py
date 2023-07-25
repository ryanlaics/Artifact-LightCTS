"""
This script serves as the training engine for the LightCTS model.

"""

import argparse, time
from engine import *
from LightCTS_model import *
import torch.nn as nn
from util import *


# Define the command line arguments that the script expects
parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cuda:0',help='device for computation')
parser.add_argument('--cts_data', type=str, default='../../data/electricity.txt',
                    help='location of the data file')
parser.add_argument('--checkpoint', type=str, default='logs/save.pt')
parser.add_argument('--horizon', type=int, default=3, help='target future horizon')
parser.add_argument('--nhid',type=float,default=24,help='embedding size')
parser.add_argument('--group',type=int,default=4,help='group number')
parser.add_argument('--nglf',type=int,default=2,help='number of GLFormer')
parser.add_argument('--batch_size',type=int,default=16,help='batch size')
parser.add_argument('--lr',type=float,default=0.0005,help='learning rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--clip',type=int,default=5,help='clip')
parser.add_argument('--epochs',type=int,default=100,help='number of training epoches')


args = parser.parse_args()
device = torch.device(args.device)
torch.set_num_threads(3)


# Load and process the dataset
data_dir = args.cts_data
Data = DataLoaderS(data_dir, 0.6, 0.2, device, args.horizon, 168)

# Initialize the training engine and the LightCTS model
engine = trainer(nhid=args.nhid, nglf=args.nglf, group=args.group, lr=args.lr, clip=args.clip, lr_decay=args.weight_decay)
model = engine.model
model = model.to(device)

# Set the loss function
criterion = nn.L1Loss(size_average=False).to(device)
evaluateL2 = nn.MSELoss(size_average=False).to(device)
evaluateL1 = nn.L1Loss(size_average=False).to(device)
best_val = 10000000


# Train the model
for epoch in range(1, args.epochs + 1):
    epoch_start_time = time.time()
    train_loss = engine.train(Data, Data.train[0], Data.train[1], model, criterion, args.batch_size)
    val_loss, val_rae, val_corr, val_mae, val_rmse = engine.eval(Data, Data.valid[0], Data.valid[1], model, evaluateL2, evaluateL1,
                                       args.batch_size)
    print(
        'Epoch: {:03d}\nTrain Loss {:5.4f}, Valid RRSE {:5.4f}, Valid CORR  {:5.4f}, Training Time: {:5.2f}s, '.format(
            epoch, train_loss, val_loss, val_corr, (time.time() - epoch_start_time)), flush=True)
    # Save the best model
    if val_loss < best_val:
        with open(args.checkpoint, 'wb') as f:
            torch.save(model, f)
        best_val = val_loss

# Load the best saved model.
with open(args.checkpoint, 'rb') as f:
    model = torch.load(f)
    
# Evaluate the model on the validation and test datasets
vtest_acc, vtest_rae, vtest_corr, vtest_mae, vtest_rmse = engine.eval(Data, Data.valid[0], Data.valid[1], model, evaluateL2, evaluateL1,
                                     args.batch_size)
test_acc, test_rae, test_corr, test_mae, test_rmse = engine.eval(Data, Data.test[0], Data.test[1], model, evaluateL2, evaluateL1,
                                     args.batch_size)
print("On average: Test RRSE: {:5.4f}, Test CORR {:5.4f}".format(test_acc, test_corr))
