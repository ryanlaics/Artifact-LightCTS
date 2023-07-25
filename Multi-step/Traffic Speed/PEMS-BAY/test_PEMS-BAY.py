
"""
This script is designed to test the lightCTS model

"""
import argparse, time, util, torch
from engine import trainer
import numpy as np

# Define the command line arguments that the script expects
parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cuda:0',help='device for computation')

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

# Define the main function where the model training and evaluation will happen
def main():
    # Set the device for computation (CPU or GPU)
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

    model.eval()
    outputs = []
    realy = []
    model.eval()
    # Iterate over the test data
    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        # Preprocess the data
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        testy = torch.Tensor(y).to(device)
        testy = testy.transpose(1, 3)[:, :1, :, :]
        with torch.no_grad():
            # Get the model's predictions
            preds = model(testx).transpose(1, 3)
        outputs.append(preds)
        realy.append(testy)

    # Concatenate the outputs of all the future time steps and compute the metrics
    yhat = torch.cat(outputs,dim=0)
    yhat = scaler.inverse_transform(yhat)
    realy = torch.cat(realy,dim=0)
    amae = []
    amape = []
    armse = []
    print(yhat.shape, realy.shape)
    for i in range(args.seq_length):
        pred = yhat[...,i]
        real = realy[...,i]
        metrics = util.metric(pred, real)
        log = 'Horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i+1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])
    log = 'On average: Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(*util.metric(yhat, realy)))

if __name__ == "__main__":
    main()
