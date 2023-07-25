"""
This script is designed to evaluate the lightness metrics of lightCTS model

"""

import argparse
from engine import *
from util import *
from fvcore.nn import FlopCountAnalysis, flop_count_table



# Define the command line arguments that the script expects
parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cpu',help='device for computation')
parser.add_argument('--cts_data', type=str, default='../../data/solar.txt',
                    help='location of the data file')
parser.add_argument('--checkpoint', type=str, default='logs/save.pt')
parser.add_argument('--horizon', type=int, default=3, help='target future horizon')
parser.add_argument('--nhid',type=float,default=32,help='embedding size')
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

# Lightness metrics
input = torch.randn(1, 1, 137, 168)
print(flop_count_table(FlopCountAnalysis(model, input)))