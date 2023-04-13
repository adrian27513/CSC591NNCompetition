import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from torch import nn
import torch
import torch.nn.functional as F

device = torch.device("cuda:0")

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

class RNN(nn.Module):
    def __init__(self, input_size, output_size, dropout, lstm_hidden, linear_hidden, lstm_layers):
        super(RNN, self).__init__()

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=lstm_hidden, num_layers=lstm_layers, batch_first=True)
        # self.batch_norm1 = nn.BatchNorm1d(lstm_hidden)
        # self.batch_norm2 = nn.BatchNorm1d(linear_hidden)\
        self.lstm_out = lstm_hidden
        self.layers = lstm_layers
        self.relu1 = nn.LeakyReLU(0.3)
        self.relu2 = nn.LeakyReLU(0.3)
        self.relu3 = nn.LeakyReLU(0.3)
        # self.relu1 = nn.ReLU()
        # self.relu2 = nn.ReLU()
        # self.relu3 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(lstm_hidden, linear_hidden)
        self.linear2 = nn.Linear(linear_hidden, linear_hidden)
        self.linear3 = nn.Linear(linear_hidden, output_size)

    def forward(self, input, state):
        # print(input.shape)
        # print(torch.zeros((1, input.shape[0], self.lstm_out)).shape)
        if state is None:
            state = (torch.zeros((self.layers, input.shape[0], self.lstm_out)).to(device=device), torch.zeros((self.layers, input.shape[0],self.lstm_out)).to(device=device))
        x, state = self.lstm(input, state)
        # print(state[0].shape)
        # print("---")
        # x = self.batch_norm1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.linear1(x)
        # x = self.batch_norm2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.linear2(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        x = self.linear3(x)
        x = x[:, -1, :]
        return x, state

# class RNN(nn.Module):
#     def __init__(self, input_size, output_size, dropout, lstm_hidden, linear_hidden, lstm_layers):
#         super(RNN, self).__init__()
#
#         self.lstm = nn.LSTM(input_size=input_size, hidden_size=lstm_hidden, num_layers=lstm_layers, batch_first=True)
#         self.dropout = nn.Dropout(dropout)
#         self.linear1 = nn.Linear(lstm_hidden, linear_hidden)
#         self.linear2 = nn.Linear(linear_hidden, output_size)
#
#     def forward(self, input):
#         x, _ = self.lstm(input)
#         x = x[:,-1,:]
#         x = self.dropout(x)
#         x = self.linear1(x)
#         x = F.relu(x)
#         x = self.linear2(x)
#         return x

class ARNN(nn.Module):
    def __init__(self, input_size, output_size, dropout, lstm_hidden, linear_hidden, lstm_layers):
        super(ARNN, self).__init__()
        self.encoder_input = nn.Linear(input_size, 512)