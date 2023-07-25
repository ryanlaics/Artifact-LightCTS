"""
This script is designed to evaluate the lightness metrics of lightCTS model

"""
import argparse, util, torch
from engine import trainer
from fvcore.nn import FlopCountAnalysis, flop_count_table

# Define the command line arguments that the script expects
parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cpu',help='device for computation')

parser.add_argument('--seq_length',type=int,default=12,help='prediction horizon')
parser.add_argument('--nhid',type=int,default=64,help='embedding size')
parser.add_argument('--in_dim',type=int,default=2,help='inputs dimension')
parser.add_argument('--batch_size',type=int,default=64,help='batch size')
parser.add_argument('--learning_rate',type=float,default=0.002,help='learning rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--epochs',type=int,default=250,help='number of training epoches')
parser.add_argument('--nglf',type=float,default=6,help='number of GLFormer')
parser.add_argument('--group',type=float,default=4,help='group number')
parser.add_argument('--print_every',type=int,default=1000,help='print frequency of training logs')
parser.add_argument('--cts_data',type=str,default='../../../data/PEMS-BAY', help='data path')
parser.add_argument('--adj_mx',type=str,default='../../../data/PEMS-BAY/adj_mx_pems_bay.pkl', help='adj data path')
parser.add_argument('--expid',type=int,default=0, help='experiment id')
parser.add_argument('--checkpoint',type=str,default=None)

args = parser.parse_args()

def main():
    # Set the device for computation (Since Fvcore only runs on CPU, we set 'cpu' by default)
    device = torch.device(args.device)
    # Get the adjacency matrix of the graph, which represents the connection of the nodes
    sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adj_mx)
    # Generate the data for model training and testing
    dataloader = util.load_dataset(args.cts_data, args.batch_size, args.batch_size, args.batch_size)
    # Get the scaler which is used to normalize the data
    scaler = dataloader['scaler']
    # Convert the adjacency matrix to tensor and move it to the device
    supports = [torch.tensor(i).to(device) for i in adj_mx]
    # Initialize the trainer engine
    engine = trainer(scaler, args.in_dim, args.seq_length, args.nhid, args.nglf, args.learning_rate, args.weight_decay,
                     args.device, supports, args.group)    # Get the model from the trainer engine
    model = engine.model
    # Move the model to the device
    model.to(device)

    # Lightness metrics
    input = torch.randn(1, 2, 325, 12)
    print(flop_count_table(FlopCountAnalysis(model, input)))

if __name__ == "__main__":
    main()

