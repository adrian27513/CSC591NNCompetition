import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from torch import nn
import torch
import torch.nn.functional as F

# class RNN(nn.Module):
#     def __init__(self, input_size, output_size, dropout, lstm_hidden, linear_hidden):
#         super(RNN, self).__init__()
#
#         self.lstm = nn.LSTM(input_size=input_size, hidden_size=lstm_hidden, num_layers=2, batch_first=True)
#         self.dropout = nn.Dropout(dropout)
#         self.linear1 = nn.Linear(lstm_hidden, linear_hidden)
#         self.linear2 = nn.Linear(linear_hidden, linear_hidden)
#         self.dropout2 = nn.Dropout(dropout)
#         self.linear3 = nn.Linear(linear_hidden, output_size)
#
#     def forward(self, input):
#         x, _ = self.lstm(input)
#         x = x[:,-1,:]
#         x = self.dropout(x)
#         x = self.linear1(x)
#         x = F.relu(x)
#         x = self.linear2(x)
#         x = F.relu(x)
#         x = self.dropout2(x)
#         x = self.linear3(x)
#         return x

# class RNN(nn.Module):
#     def __init__(self, input_size, output_size, dropout, lstm_hidden, linear_hidden):
#         super(RNN, self).__init__()
#
#         self.lstm = nn.LSTM(input_size=input_size, hidden_size=lstm_hidden, num_layers=1, batch_first=True)
#         self.batch_norm1 = nn.BatchNorm1d(lstm_hidden)
#         self.dropout = nn.Dropout(dropout)
#         self.linear1 = nn.Linear(lstm_hidden, linear_hidden)
#         self.batch_norm2 = nn.BatchNorm1d(linear_hidden)
#         self.linear2 = nn.Linear(linear_hidden, output_size)
#
#     def forward(self, input):
#         x, _ = self.lstm(input)
#         x = x[:,-1,:]
#         x = self.batch_norm1(x)
#         x = F.relu(x)
#         x = self.dropout(x)
#         x = self.linear1(x)
#         x = self.batch_norm2(x)
#         x = F.relu(x)
#         x = self.dropout(x)
#         x = self.linear2(x)
#         return x

class RNN(nn.Module):
    def __init__(self, input_size, output_size, dropout, lstm_hidden, linear_hidden, lstm_layers):
        super(RNN, self).__init__()

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=lstm_hidden, num_layers=lstm_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(lstm_hidden, linear_hidden)
        self.linear2 = nn.Linear(linear_hidden, output_size)

    def forward(self, input):
        x, _ = self.lstm(input)
        x = x[:,-1,:]
        x = self.dropout(x)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x
