
import os
import sys
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
import torch.nn.init as init
from torch import autograd
from torch.autograd import Variable
from utils import *


EPOCHS=3
BATCHSIZE=64
EMBEDSIZE=125
NUMHIDDEN=100
DROPOUT=0.2
LR=0.001
BETA_1=0.9
BETA_2=0.999
EPS=1e-08
MAXLEN=150 #maximum size of the word sequence
MAXFEATURES=30000 #vocabulary size
GPU=True

class MyNet(nn.Module):
    def __init__(self, 
                 maxf=MAXFEATURES, edim=EMBEDSIZE, nhid=NUMHIDDEN):
        super(MyNet, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=maxf,
                                      embedding_dim=edim)
        # If batch-first then input and output 
        # provided as (batch, seq, features)
        # Cudnn used by default if possible
        self.gru = nn.GRU(input_size=edim, 
                          hidden_size=nhid, 
                          num_layers=1,
                          batch_first=True,
                          bidirectional=False)   
        self.l_out = nn.Linear(in_features=nhid*1,
                               out_features=2)

    def forward(self, x, nhid=NUMHIDDEN, batchs=BATCHSIZE):
        x = self.embedding(x.long())
        h0 = Variable(torch.zeros(1, batchs, nhid)).cuda()
        x, h = self.gru(x, h0)  # outputs, states
        # just get the last output state
        x = x[:,-1,:].squeeze()
        x = self.l_out(x)
        return x

def init_model(m, lr=LR, b1=BETA_1, b2=BETA_2, eps=EPS):
    opt = optim.Adam(m.parameters(), lr, betas=(b1, b2), eps=eps)
    criterion = nn.CrossEntropyLoss()
    return opt, criterion

if __name__ == '__main__':

    # Data into format for library
    x_train, x_test, y_train, y_test = imdb_for_library(seq_len=MAXLEN, max_features=MAXFEATURES)
    # Torch-specific
    x_train = x_train.astype(np.int64)
    x_test = x_test.astype(np.int64)
    y_train = y_train.astype(np.int64)
    y_test = y_test.astype(np.int64)
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    print(x_train.dtype, x_test.dtype, y_train.dtype, y_test.dtype)


    mod = MyNet().cuda()
    optimizer, criterion = init_model(mod)

    mod.train() # Sets training = True   
    for j in range(EPOCHS):
        for data, target in yield_mb(x_train, y_train, BATCHSIZE, shuffle=True):
            # Get samples
            data = torch.tensor(data, requires_grad = True, dtype=torch.float).cuda() #Variable(torch.LongTensor(data).cuda())
            target = torch.tensor(target, dtype=torch.long).cuda()
            # Init
            optimizer.zero_grad()
            # Forwards
            output = mod(data)
            # Loss
            loss = criterion(output, target)
            # Back-prop
            loss.backward()
            optimizer.step()
        # Log
        print(j)

    mod.eval() # Sets training = False
    n_samples = (y_test.shape[0]//BATCHSIZE)*BATCHSIZE
    y_guess = np.zeros(n_samples, dtype=np.int)
    y_truth = y_test[:n_samples]
    c = 0
    for data, target in yield_mb(x_test, y_test, BATCHSIZE):
        # Get samples
        data = Variable(torch.LongTensor(data).cuda())
        # Forwards
        output = mod(data)
        pred = output.data.max(1)[1].cpu().numpy().squeeze()
        # Collect results
        y_guess[c*BATCHSIZE:(c+1)*BATCHSIZE] = pred
        c += 1


    print("Accuracy: ", sum(y_guess == y_truth)/len(y_guess))

