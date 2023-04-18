import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
from torch import nn

device = torch.device("cuda:0")

class Decoder(nn.Module):
    def __init__(self, dec_units):
        super(Decoder, self).__init__()
        self.dec_units = dec_units
        self.leaky = 0.2

        self.W1 = nn.Linear(6, dec_units)
        self.W2 = nn.Linear(dec_units, dec_units)
        self.V = nn.Linear(dec_units + 1, 1)
        self.batch1 = nn.BatchNorm1d(1)


        self.cnn1 = nn.Conv1d(kernel_size=11, in_channels=6, out_channels=6)
        self.cnn2 = nn.Conv1d(kernel_size=11, in_channels=6, out_channels=6)

        self.max1 = nn.MaxPool1d(kernel_size=2)

        self.cnn3 = nn.Conv1d(kernel_size=5, in_channels=6, out_channels=6)
        self.cnn4 = nn.Conv1d(kernel_size=5, in_channels=6, out_channels=6)

        self.max2 = nn.MaxPool1d(kernel_size=2)
        self.batch2 = nn.BatchNorm1d(6)

        self.gru = nn.GRU(input_size=6, hidden_size=dec_units)

        self.linear1 = nn.Linear(dec_units, 400)
        self.batch3 = nn.BatchNorm1d(1)
        self.linear2 = nn.Linear(400, 1)
        self.linear3 = nn.Linear(int(((((dec_units - 20)/2)-8)/2) + 4), 50)
        self.linear4 = nn.Linear(50, 4)

    def forward(self, last_input, hidden, enc_output):
        # print(last_input.shape)
        # print(hidden.shape)
        # print(enc_output.shape)
        w1 = self.W1(enc_output)
        w1 = F.leaky_relu(w1, self.leaky)

        w2 = self.W2(hidden).permute(2,1,0)
        w2 = F.leaky_relu(w2, self.leaky)

        # print(w1.shape)
        # print(w2.shape)

        cat = torch.cat((w1, w2), dim=-1)
        # print(cat.shape)
        score = self.V(cat)
        score = self.batch1(score)
        score = F.leaky_relu(score, self.leaky)
        # print(score.shape)
        # print(score)
        weights = F.softmax(score, dim=0)
        # print(weights)
        # print(weights.shape)
        # print(weights)
        # print(enc_output)
        context = enc_output * weights
        # print(context.shape)
        context = context.permute(1, 2, 0)

        context = self.cnn1(context)
        context = F.leaky_relu(context, self.leaky)
        context = self.cnn2(context)
        context = F.leaky_relu(context, self.leaky)
        # print(context.shape)
        context = self.max1(context)
        # print(context.shape)
        context = self.cnn3(context)
        context = F.leaky_relu(context, self.leaky)
        context = self.cnn4(context)
        # print(context.shape)
        context = F.leaky_relu(context, self.leaky)
        context = self.batch2(context)
        context = self.max2(context)

        # print(context.shape)

        context = context.permute(2,0,1)
        # print(context.shape)
        x, hidden = self.gru(context, hidden)
        x = F.leaky_relu(x, self.leaky)
        # print(x.shape)
        # print(hidden.shape)

        x = self.linear1(x)
        # print(x.shape)
        x = self.batch3(x)
        x = F.leaky_relu(x, self.leaky)
        x = F.dropout(x, 0.5)
        # print(x.shape)
        x = self.linear2(x).squeeze()
        x = F.leaky_relu(x, self.leaky)
        x = F.dropout(x, 0.5)
        # print(x.shape)

        x = torch.cat((x, last_input))

        x = self.linear3(x)
        x = F.leaky_relu(x, self.leaky)
        x = F.dropout(x, 0.5)
        # print(x.shape)

        x = self.linear4(x)
        # print(x.shape)
        # print(hidden.shape)
        return x, hidden

    def initialize_hidden(self, window):
        return torch.zeros((1,window,self.dec_units))