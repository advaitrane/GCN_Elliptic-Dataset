import numpy as np
import time
import sys
import pandas as pd
import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import GCN_2layer
from utils import load_data


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_dir', type = str, default = "elliptic_bitcoin_dataset", 
					help = "Dataset directory which contains thre csv files")
parser.add_argument('-e', '--epochs', type = int, default = 50, 
					help = "Number of epochs to train each timestep on")
parser.add_argument('-l', '--lr', type = int, default = 0.001, 
					help = "Learning rate for adam optimizer")
parser.add_argument('-hs', '--hidden_layer', type = int, default = 16, 
					help = "Size of the hidden layer")
parser.add_argument('-t', '--train_ts', type = int, default = 34, 
					help = "Number of timesteps to train on")
parser.add_argument('-m', '--model_dir', type = str, default = "gcn_weights",
					help = "Directory to store the weights after training")
arg = parser.parse_args()


DATA_DIR = arg.data_dir
MODEL_DIR = arg.model_dir

try:
    os.mkdir(MODEL_DIR)
except FileExistsError:
    pass

num_features = 166
num_hidden_features = arg.hidden_layer
num_classes = 2
num_ts = 49
epochs = arg.epochs
lr = arg.lr
max_train_ts = arg.train_ts
train_ts = np.arange(max_train_ts)

adj_mats, features_labelled_ts, classes_ts = load_data(DATA_DIR, 0, max_train_ts)

# 0 - illicit, 1 - licit
labels_ts = []
for c in classes_ts:
    labels_ts.append(np.array(c['class'] == '2', dtype = np.long))

skip_gcn = GCN_2layer(num_features, num_hidden_features, num_classes, skip = True)
skip_train_loss = nn.CrossEntropyLoss(weight = torch.DoubleTensor([0.7, 0.3]))
skip_optimizer = torch.optim.Adam(skip_gcn.parameters(), lr = lr)

for ts in train_ts:
    A = torch.tensor(adj_mats[ts].values)
    X = torch.tensor(features_labelled_ts[ts].values)
    L = torch.tensor(labels_ts[ts], dtype = torch.long)
    for ep in range(epochs):
        t_start = time.time()
        
        skip_gcn.train()
        skip_optimizer.zero_grad()
        skip_out = skip_gcn(A, X)

        skip_loss = skip_train_loss(skip_out, L)
        skip_train_pred = skip_out.max(1)[1].type_as(L)
        skip_acc = (skip_train_pred.eq(L).double().sum())/L.shape[0]

        skip_loss.backward()
        skip_optimizer.step()

        sys.stdout.write("\r Epoch %d/%d Timestamp %d/%d training loss: %f training accuracy: %f Time: %s"
                         %(ep, epochs, ts, max_train_ts, skip_loss, skip_acc, time.time() - t_start)
                        )

torch.save(skip_gcn.state_dict(), str(os.path.join(MODEL_DIR, "skip_gcn_weights.pth")))

