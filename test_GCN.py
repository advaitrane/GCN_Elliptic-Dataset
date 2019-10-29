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

from sklearn.metrics import f1_score, precision_score, recall_score


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_dir', type = str, default = "elliptic_bitcoin_dataset", 
					help = "Dataset directory which contains thre csv files")
parser.add_argument('-t', '--test_ts', type = int, default = 34, 
					help = "Number of the timestep to start testing from")
parser.add_argument('-m', '--model_dir', type = str, default = "gcn_weights",
					help = "Directory to store the weights after training")
arg = parser.parse_args()


DATA_DIR = arg.data_dir
MODEL_DIR = arg.model_dir

num_features = 166
num_classes = 2
num_ts = 49
min_test_ts = arg.test_ts
test_ts = np.arange(num_ts - min_test_ts)

adj_mats, features_labelled_ts, classes_ts = load_data(DATA_DIR, min_test_ts, num_ts)

# 0 - illicit, 1 - licit
labels_ts = []
for c in classes_ts:
    labels_ts.append(np.array(c['class'] == '2', dtype = np.long))

gcn = GCN_2layer(num_features, 32, num_classes)
gcn.load_state_dict(torch.load(os.path.join(MODEL_DIR, "gcn_weights.pth")))

# Testing
test_accs = []
test_precisions = []
test_recalls = []
test_f1s = []

for ts in test_ts:
    A = torch.tensor(adj_mats[ts].values)
    X = torch.tensor(features_labelled_ts[ts].values)
    L = torch.tensor(labels_ts[ts], dtype = torch.long)
    
    gcn.eval()
    test_out = gcn(A, X)
    
    test_pred = test_out.max(1)[1].type_as(L)
    t_acc = (test_pred.eq(L).double().sum())/L.shape[0]
    test_accs.append(t_acc.item())
    test_precisions.append(precision_score(L, test_pred))
    test_recalls.append(recall_score(L, test_pred))
    test_f1s.append(f1_score(L, test_pred))

acc = np.array(test_accs).mean()
prec = np.array(test_precisions).mean()
rec = np.array(test_recalls).mean()
f1 = np.array(test_f1s).mean()

print("GCN - averaged accuracy: {}, precision: {}, recall: {}, f1: {}".format(acc, prec, rec, f1))