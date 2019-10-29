import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import time
import sys

# class to define a Graph Convolutional Layer

class GraphConv(nn.Module):
    def __init__(self, in_features, out_features, activation  = 'relu', skip = False, skip_in_features = None):
        super(GraphConv, self).__init__()
        self.W = torch.nn.Parameter(torch.DoubleTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.W)
        
        self.set_act = False
        if activation == 'relu':
            self.activation = nn.ReLU()
            self.set_act = True
        elif activation == 'softmax':
            self.activation = nn.Softmax(dim = 1)
            self.set_act = True
        else:
            self.set_act = False
            raise ValueError("activations supported are 'relu' and 'softmax'")
            
        self.skip = skip
        if self.skip:
            if skip_in_features == None:
                raise ValueError("pass input feature size of the skip connection")
            self.W_skip = torch.nn.Parameter(torch.DoubleTensor(skip_in_features, out_features)) 
            nn.init.xavier_uniform_(self.W)
        
    def forward(self, A, H_in, H_skip_in = None):
        # A must be an n x n matrix as it is an adjacency matrix
        # H is the input of the node embeddings, shape will n x in_features
        self.A = A
        self.H_in = H_in
        A_ = torch.add(self.A, torch.eye(self.A.shape[0]).double())
        D_ = torch.diag(A_.sum(1))
        # since D_ is a diagonal matrix, 
        # its root will be the roots of the diagonal elements on the principle diagonal
        # since A is an adjacency matrix, we are only dealing with positive values 
        # all roots will be real
        D_root_inv = torch.inverse(torch.sqrt(D_))
        A_norm = torch.mm(torch.mm(D_root_inv, A_), D_root_inv)
        # shape of A_norm will be n x n
        
        H_out = torch.mm(torch.mm(A_norm, H_in), self.W)
        # shape of H_out will be n x out_features
        
        if self.skip:
            H_skip_out = torch.mm(H_skip_in, self.W_skip)
            H_out = torch.add(H_out, H_skip_out)
        
        if self.set_act:
            H_out = self.activation(H_out)
            
        return H_out
        

# class for 2 layer Graph Convolutional Network

class GCN_2layer(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, skip = False):
        super(GCN_2layer, self).__init__()
        self.skip = skip
        
        self.gcl1 = GraphConv(in_features, hidden_features)
        
        if self.skip:
            self.gcl_skip = GraphConv(hidden_features, out_features, activation = 'softmax', skip = self.skip,
                                  skip_in_features = in_features)
        else:
            self.gcl2 = GraphConv(hidden_features, out_features, activation = 'softmax')
        
    def forward(self, A, X):
        out = self.gcl1(A, X)
        if self.skip:
            out = self.gcl_skip(A, out, X)
        else:
            out = self.gcl2(A, out)
            
        return out