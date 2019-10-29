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

gcn = GCN_2layer(num_features, 32, num_classes)
train_loss = nn.CrossEntropyLoss(weight = torch.DoubleTensor([0.7, 0.3]))
optimizer = torch.optim.Adam(gcn.parameters(), lr = lr)

# Training

for ts in train_ts:
    A = torch.tensor(adj_mats[ts].values)
    X = torch.tensor(features_labelled_ts[ts].values)
    L = torch.tensor(labels_ts[ts], dtype = torch.long)
    for ep in range(epochs):
        t_start = time.time()
        
        gcn.train()
        optimizer.zero_grad()
        out = gcn(A, X)

        loss = train_loss(out, L)
        train_pred = out.max(1)[1].type_as(L)
        acc = (train_pred.eq(L).double().sum())/L.shape[0]

        loss.backward()
        optimizer.step()

        sys.stdout.write("\r Epoch %d/%d Timestamp %d/%d training loss: %f training accuracy: %f Time: %s"
                         %(ep, epochs, ts, max_train_ts, loss, acc, time.time() - t_start)
                        )

torch.save(gcn.state_dict(), str(os.path.join(MODEL_DIR, "gcn_weights.pth")))

