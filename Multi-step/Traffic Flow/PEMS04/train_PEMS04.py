"""
This script is designed to train the lightCTS model

"""
import argparse, time, util, torch
from engine import trainer
import numpy as np

# Define the command line arguments that the script expects
parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cuda:3',help='device for computation')
parser.add_argument('--seq_length',type=int,default=12,help='prediction horizon')
parser.add_argument('--nhid',type=int,default=64,help='embedding size')
parser.add_argument('--in_dim',type=int,default=1,help='inputs dimension')
parser.add_argument('--batch_size',type=int,default=64,help='batch size')
parser.add_argument('--learning_rate',type=float,default=0.002,help='learning rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--epochs',type=int,default=200,help='number of training epoches')
parser.add_argument('--nglf',type=float,default=4,help='number of GLFormer')
parser.add_argument('--group',type=float,default=4,help='group number')
parser.add_argument('--print_every',type=int,default=1000,help='print frequency of training logs')
parser.add_argument('--save',type=str,default='logs/',help='save path')
parser.add_argument('--adj_mx',type=str,default='../../../data/PEMS04/PEMS04.csv',help='path to the spatial adjacency matrix')
parser.add_argument('--cts_data',type=str,default='../../../data/PEMS04/PEMS04.npz',help='path to the CTS data')
parser.add_argument('--expid',type=int,default=0,help='experiment id')
args = parser.parse_args()

# Define the main function where the model training and evaluation will happen
def main():
    # Set the device for computation
    device = torch.device(args.device)
    # Get the adjacency matrix of the graph, which represents the connection of the nodes
    adj_mx = util.get_adj_matrix(args.adj_mx, 307)
    # Generate the data for model training and testing
    dataloader = util.generate_data(args.cts_data, args.batch_size, args.batch_size)
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

    # Initialize some lists to store the loss and time
    his_loss =[]
    val_time = []
    train_time = []
    # Start training for a number of epochs
    for i in range(1, args.epochs+1):
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        # Iterate over the training data
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator(),start=1):
            trainx = torch.Tensor(x).to(device)
            trainx= trainx.transpose(1, 3)
            trainy = torch.Tensor(y).to(device)
            trainy = trainy.transpose(1, 3)
            metrics = engine.train(trainx, trainy[:,0,:,:])
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            if iter % args.print_every == 0:
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]),flush=True)
        t2 = time.time()
        train_time.append(t2-t1)
        valid_loss = []
        valid_mape = []
        valid_rmse = []
        s1 = time.time()
        # Iterate over the validation data
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3)
            metrics = engine.eval(testx, testy[:,0,:,:])
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i,(s2-s1)))
        val_time.append(s2-s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        his_loss.append(mvalid_loss)

        log = 'Epoch: {:03d}\nTrain Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}\nValid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}\nTraining Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)),flush=True)
        torch.save(engine.model.state_dict(), args.save+"_epoch_"+str(i)+"_"+str(round(mvalid_loss,2))+".pth")

    # Load the model with the smallest validation loss
    bestid = np.argmin(his_loss)
    engine.model.load_state_dict(torch.load(args.save+"_epoch_"+str(bestid+1)+"_"+str(round(his_loss[bestid],2))+".pth"))
    engine.model.eval()

    outputs = []
    realy = []
    # Iterate over the test data
    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        # Preprocess the data
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1,3)
        testy = torch.Tensor(y).to(device)
        testy = testy.transpose(1,3)[:,:1,:,:]
        with torch.no_grad():
            # Get the model's predictions
            preds = engine.model(testx).transpose(1,3)
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

    # Save the best model's parameters
    torch.save(engine.model.state_dict(), args.save+"_exp"+str(args.expid)+"_best_"+str(round(his_loss[bestid],2))+".pth")

if __name__ == "__main__":
    main()
