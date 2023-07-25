"""
This script contains several utility functions and classes for data processing.

The main components are:

1. DataLoader: A class for loading data in batches.

2. StandardScaler: A class for standardizing data by removing the mean and scaling to unit variance.

3. load_dataset: A function that loads train, validation, and test datasets from a specified directory.

4. Metric calculation functions: These include 'masked_mse', 'masked_rmse', 'masked_mae',
   'masked_mape', and 'metric'. These functions are used to calculate various metrics for the evaluation of model performance.

5. generate_data: A function that loads the data and generates train, validation, and test datasets.
   It also standardizes the data and shuffles it, creating data loaders for each dataset.

6. generate_from_train_val_test and generate_from_data: These functions generate sequences of data
   from the provided train, val, and test sets or split the data into train, val, and test sets and
   generate sequences of data from each set.

7. generate_seq: A function that generates sequences of data for a given train length and prediction length.

8. get_adj_matrix: A function that generates an adjacency matrix for the data.
"""


import pickle, os, torch, csv, copy
import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg
import torch.nn as nn

# This class is used to load data in batches.
class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        self.batch_size = batch_size
        self.current_ind = 0
        # Use the last sample of the data to fill up the last batch if the size of data is not a multiple of batch size.
        num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
        x_padding = np.repeat(xs[-1:], num_padding, axis=0)
        y_padding = np.repeat(ys[-1:], num_padding, axis=0)
        xs = np.concatenate([xs, x_padding], axis=0)
        ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    # This method is used to shuffle the data.
    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    # This method is used to get an iterator for the batches.
    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()

# This class is used for standardizing the data.
class StandardScaler():

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    # This method is used to standardize the data.
    def transform(self, data):
        return (data - self.mean) / self.std

    # This method is used to undo the standardization.
    def inverse_transform(self, data):
        return (data * self.std) + self.mean

# This function is used to load the dataset.
def load_dataset(dataset_dir, batch_size, valid_batch_size= None, test_batch_size=None):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']


    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], valid_batch_size)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size)
    data['scaler'] = scaler
    return data

# The following four functions are used to calculate different metrics.
def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))

def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

# Calculate and return MAE, MAPE, and RMSE metrics
def metric(pred, real):
    mae = masked_mae(pred,real,0.0).item()
    mape = masked_mape(pred,real,0.0).item()
    rmse = masked_rmse(pred,real,0.0).item()
    return mae, mape, rmse
def generate_data(graph_signal_matrix_name, batch_size, test_batch_size=None, transformer=None):
    # This function loads the data and generates train, validation, and test datasets.
    origin_data = np.load(graph_signal_matrix_name)  # Load the data from .npz file.
    keys = origin_data.keys()
    if 'train' in keys and 'val' in keys and 'test' in keys:
        # If the data file has already been split into train, val, and test sets,
        # generate data from these sets.
        data = generate_from_train_val_test(origin_data, transformer)
    elif 'data' in keys:
        # If the data file is not split, split it into train, val, and test sets.
        length = origin_data['data'].shape[0]
        data = generate_from_data(origin_data, length, transformer)
    else:
        raise KeyError("neither data nor train, val, test is in the data")

    # Standardize the data.
    scalar = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scalar.transform(data['x_' + category][..., 0])

    # Shuffle the data and generate data loaders.
    train_len = len(data['x_train'])
    permutation = np.random.permutation(train_len)
    data['x_train_1'] = data['x_train'][permutation][:int(train_len / 2)]
    data['y_train_1'] = data['y_train'][permutation][:int(train_len / 2)]
    data['x_train_2'] = data['x_train'][permutation][int(train_len / 2):]
    data['y_train_2'] = data['y_train'][permutation][int(train_len / 2):]
    data['x_train_3'] = copy.deepcopy(data['x_train_2'])
    data['y_train_3'] = copy.deepcopy(data['y_train_2'])
    data['train_loader_1'] = DataLoader(data['x_train_1'], data['y_train_1'], batch_size)
    data['train_loader_2'] = DataLoader(data['x_train_2'], data['y_train_2'], batch_size)
    data['train_loader_3'] = DataLoader(data['x_train_3'], data['y_train_3'], batch_size)
    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], test_batch_size)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size)
    data['scaler'] = scalar

    return data


def generate_from_train_val_test(origin_data, transformer):
    # This function generates sequences of data from the provided train, val, and test sets.
    data = {}
    for key in ('train', 'val', 'test'):
        x, y = generate_seq(origin_data[key], 12, 12)
        data['x_' + key] = x.astype('float32')
        data['y_' + key] = y.astype('float32')

    return data


def generate_from_data(origin_data, length, transformer):
    # This function splits the data into train, val, and test sets and generates sequences of data from each set.
    data = {}
    train_line, val_line = int(length * 0.6), int(length * 0.8)
    for key, line1, line2 in (('train', 0, train_line),
                              ('val', train_line, val_line),
                              ('test', val_line, length)):

        x, y = generate_seq(origin_data['data'][line1: line2], 12, 12)
        data['x_' + key] = x.astype('float32')
        data['y_' + key] = y.astype('float32')

    return data


def generate_seq(data, train_length, pred_length):
    # This function generates sequences of data for a given train length and prediction length.
    seq = np.concatenate([np.expand_dims(
        data[i: i + train_length + pred_length], 0)
        for i in range(data.shape[0] - train_length - pred_length + 1)],
        axis=0)[:, :, :, 0: 1]
    return np.split(seq, 2, axis=1)

def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def load_adj(pkl_filename):
    # This function loads an adjacency matrix for the data.
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
    return sensor_ids, sensor_id_to_ind, adj


def get_adj_matrix(distance_df_filename, num_of_vertices, type_='connectivity', id_filename=None):
    # This function generates an adjacency matrix for the data.
    A = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)
    if id_filename:
        # If there is an id file, use it to map the ids to indices.
        with open(id_filename, 'r') as f:
            id_dict = {int(i): idx
                       for idx, i in enumerate(f.read().strip().split('\n'))}
        with open(distance_df_filename, 'r') as f:
            f.readline()
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 3:
                    continue
                i, j, distance = int(row[0]), int(row[1]), float(row[2])
                A[id_dict[i], id_dict[j]] = 1
                A[id_dict[j], id_dict[i]] = 1
        return A

    # If there is no id file, generate the adjacency matrix directly from the distance file.
    with open(distance_df_filename, 'r') as f:
        f.readline()
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 3:
                continue
            i, j, distance = int(row[0]), int(row[1]), float(row[2])
            if type_ == 'connectivity':
                A[i, j] = 1
                A[j, i] = 1
            elif type_ == 'distance':
                A[i, j] = 1 / distance
                A[j, i] = 1 / distance
            else:
                raise ValueError("type_ error, must be connectivity or distance!")

    return A