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
        self.cnn2 = nn.Conv1d(kernel_size=11, in_channels=6, out_channels=12)

        self.max1 = nn.MaxPool1d(kernel_size=2)

        self.cnn3 = nn.Conv1d(kernel_size=5, in_channels=12, out_channels=12)
        self.cnn4 = nn.Conv1d(kernel_size=5, in_channels=12, out_channels=24)

        self.max2 = nn.MaxPool1d(kernel_size=2)

        self.cnn5 = nn.Conv1d(kernel_size=3, in_channels=24, out_channels=24)
        self.cnn6 = nn.Conv1d(kernel_size=3, in_channels=24, out_channels=24)

        self.max3 = nn.MaxPool1d(kernel_size=2)

        self.batch2 = nn.BatchNorm1d(24)

        self.gru = nn.GRU(input_size=24, hidden_size=dec_units)

        cnn1 = (dec_units - 20) / 2
        cnn2 = (cnn1 - 8) / 2
        cnn3 = (cnn2 - 4) / 2

        self.linear1 = nn.Linear(int(cnn3) * dec_units, 400)
        self.batch3 = nn.BatchNorm1d(int(cnn3) * dec_units)
        self.linear2 = nn.Linear(400, 400)
        self.linear3 = nn.Linear(404, 100)
        self.batch4 = nn.BatchNorm1d(100)
        self.linear4 = nn.Linear(100, 50)
        self.linear5 = nn.Linear(50, 4)

    def forward(self, last_input, hidden, enc_output):
        # print(last_input.shape)
        # print(hidden.shape)
        # print(enc_output.shape)
        w1 = self.W1(enc_output)
        w1 = F.leaky_relu(w1, self.leaky)
        w1 = F.dropout(w1, 0.5)

        w2 = self.W2(hidden).permute(2, 1, 0)
        w2 = F.leaky_relu(w2, self.leaky)
        w2 = F.dropout(w2, 0.5)

        # print(w1.shape)
        # print(w2.shape)

        cat = torch.cat((w1, w2), dim=-1)
        # print(cat.shape)
        score = self.V(cat)
        score = self.batch1(score)
        score = F.leaky_relu(score, self.leaky)
        # score = F.dropout(score, 0.5)
        # print(score.shape)
        # print(score)
        weights = F.softmax(score, dim=0)
        # print(weights.shape)
        # print(weights)
        # print(enc_output)
        context = enc_output * weights
        context = F.leaky_relu(context, self.leaky)
        context = F.dropout(context, 0.5)
        # print(context.shape)
        context = context.permute(1, 2, 0)

        context = self.cnn1(context)
        context = F.leaky_relu(context, self.leaky)
        context = F.dropout(context, 0.1)
        context = self.cnn2(context)
        context = F.leaky_relu(context, self.leaky)
        context = F.dropout(context, 0.1)
        context = self.max1(context)

        context = self.cnn3(context)
        context = F.leaky_relu(context, self.leaky)
        context = F.dropout(context, 0.15)
        context = self.cnn4(context)
        context = F.leaky_relu(context, self.leaky)
        context = F.dropout(context, 0.15)
        context = self.batch2(context)
        context = self.max2(context)

        context = self.cnn5(context)
        context = F.leaky_relu(context, self.leaky)
        context = F.dropout(context, 0.2)
        context = self.cnn6(context)
        context = F.leaky_relu(context, self.leaky)
        context = F.dropout(context, 0.2)
        context = self.max3(context)

        # context = torch.flatten(context)
        # print(context.shape)

        context = context.permute(2, 0, 1)
        # print(context.shape)
        x, hidden = self.gru(context, hidden)
        hidden = F.tanh(hidden)
        x = F.leaky_relu(x, self.leaky)
        context = F.dropout(context, 0.5)
        # print(x.shape)
        # print(hidden.shape)
        x = torch.flatten(x)

        x = self.linear1(x)
        x = F.leaky_relu(x, self.leaky)
        x = F.dropout(x, 0.5)
        # print(x.shape)
        x = self.linear2(x)
        x = F.leaky_relu(x, self.leaky)
        x = F.dropout(x, 0.5)
        # print(x.shape)

        x = torch.cat((x, last_input))

        x = self.linear3(x)
        x = F.leaky_relu(x, self.leaky)
        x = F.dropout(x, 0.5)
        # print(x.shape)
        # x = self.batch4(x)
        x = self.linear4(x)
        x = F.leaky_relu(x, self.leaky)
        x = F.dropout(x, 0.2)

        x = self.linear5(x)
        # print(x.shape)
        # print(hidden.shape)
        return x, hidden

    def initialize_hidden(self, window):
        return torch.zeros((1, window, self.dec_units))