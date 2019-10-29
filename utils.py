import numpy as np
import time
import sys
import pandas as pd
import os

def load_data(data_dir, start_ts, end_ts):
	classes_csv = 'elliptic_txs_classes.csv'
	edgelist_csv = 'elliptic_txs_edgelist.csv'
	features_csv = 'elliptic_txs_features.csv'

	classes = pd.read_csv(os.path.join(data_dir, classes_csv), index_col = 'txId') # labels for the transactions i.e. 'unknown', '1', '2'
	edgelist = pd.read_csv(os.path.join(data_dir, edgelist_csv), index_col = 'txId1') # directed edges between transactions
	features = pd.read_csv(os.path.join(data_dir, features_csv), header = None, index_col = 0) # features of the transactions
	
	num_features = features.shape[1]
	num_tx = features.shape[0]	
	total_tx = list(classes.index)

	# select only the transactions which are labelled
	labelled_classes = classes[classes['class'] != 'unknown']
	labelled_tx = list(labelled_classes.index)

	# to calculate a list of adjacency matrices for the different timesteps

	adj_mats = []
	features_labelled_ts = []
	classes_ts = []
	num_ts = 49 # number of timestamps from the paper

	for ts in range(start_ts, end_ts):
	    features_ts = features[features[1] == ts+1]
	    tx_ts = list(features_ts.index)
	    
	    labelled_tx_ts = [tx for tx in tx_ts if tx in set(labelled_tx)]
	    
	    # adjacency matrix for all the transactions
	    # we will only fill in the transactions of this timestep which have labels and can be used for training
	    adj_mat = pd.DataFrame(np.zeros((num_tx, num_tx)), index = total_tx, columns = total_tx)
	    
	    edgelist_labelled_ts = edgelist.loc[edgelist.index.intersection(labelled_tx_ts).unique()]
	    for i in range(edgelist_labelled_ts.shape[0]):
	        adj_mat.loc[edgelist_labelled_ts.index[i], edgelist_labelled_ts.iloc[i]['txId2']] = 1
	    
	    adj_mat_ts = adj_mat.loc[labelled_tx_ts, labelled_tx_ts]
	    features_l_ts = features.loc[labelled_tx_ts]
	    
	    adj_mats.append(adj_mat_ts)
	    features_labelled_ts.append(features_l_ts)
	    classes_ts.append(classes.loc[labelled_tx_ts])

	return adj_mats, features_labelled_ts, classes_ts

	