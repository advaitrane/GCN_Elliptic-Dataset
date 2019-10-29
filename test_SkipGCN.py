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
parser.add_argument('-hs', '--hidden_layer', type = int, default = 16, 
					help = "Size of the hidden layer")
parser.add_argument('-m', '--model_dir', type = str, default = "gcn_weights",
					help = "Directory to store the weights after training")
arg = parser.parse_args()


DATA_DIR = arg.data_dir
MODEL_DIR = arg.model_dir

num_features = 166
num_hidden_features = arg.hidden_layer
num_classes = 2
num_ts = 49
min_test_ts = arg.test_ts
test_ts = np.arange(num_ts - min_test_ts)

adj_mats, features_labelled_ts, classes_ts = load_data(DATA_DIR, min_test_ts, num_ts)

# 0 - illicit, 1 - licit
labels_ts = []
for c in classes_ts:
    labels_ts.append(np.array(c['class'] == '2', dtype = np.long))

skip_gcn = GCN_2layer(num_features, num_hidden_features, num_classes, skip = True)
skip_gcn.load_state_dict(torch.load(os.path.join(MODEL_DIR, "skip_gcn_weights.pth")))

skip_test_accs = []
skip_test_precisions = []
skip_test_recalls = []
skip_test_f1s = []

for ts in test_ts:
    A = torch.tensor(adj_mats[ts].values)
    X = torch.tensor(features_labelled_ts[ts].values)
    L = torch.tensor(labels_ts[ts], dtype = torch.long)
    
    skip_gcn.eval()
    skip_test_out = skip_gcn(A, X)
    
    skip_test_pred = skip_test_out.max(1)[1].type_as(L)
    skip_t_acc = (skip_test_pred.eq(L).double().sum())/L.shape[0]
    skip_test_accs.append(skip_t_acc.item())
    skip_test_precisions.append(precision_score(L, skip_test_pred))
    skip_test_recalls.append(recall_score(L, skip_test_pred))
    skip_test_f1s.append(f1_score(L, skip_test_pred))

skip_acc = np.array(skip_test_accs).mean()
skip_prec = np.array(skip_test_precisions).mean()
skip_rec = np.array(skip_test_recalls).mean()
skip_f1 = np.array(skip_test_f1s).mean()

print("SkipGCN - averaged accuracy: {}, precision: {}, recall: {}, f1: {}".format(skip_acc, skip_prec, skip_rec, skip_f1))