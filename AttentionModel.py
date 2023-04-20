import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
from torch import nn

device = torch.device("cuda:0")


class AttentionModule(nn.Module):
    def __init__(self):
        super(AttentionModule, self).__init__()
        self.leaky = 0.2

        self.attention1 = nn.MultiheadAttention(6, 2)
        self.layer_norm1 = nn.LayerNorm(6)

        self.attention2 = nn.MultiheadAttention(6, 3)
        self.layer_norm2 = nn.LayerNorm(6)

        self.att_linear1 = nn.Linear(6, 50)
        self.att_linear2 = nn.Linear(50, 6)
        self.layer_norm3 = nn.LayerNorm(6)

    def forward(self, enc_output):
        att1, _ = self.attention1(query=enc_output, key=enc_output, value=enc_output)
        att1 = F.dropout(att1, 0.5)
        add1 = torch.add(att1, enc_output)
        add1 = F.leaky_relu(add1, self.leaky)
        norm1 = self.layer_norm1(add1)

        att2, _ = self.attention2(query=enc_output, key=enc_output, value=norm1)
        att2 = F.dropout(att2, 0.5)
        add2 = torch.add(att2, norm1)
        add2 = F.leaky_relu(add2, self.leaky)
        norm2 = self.layer_norm2(add2)

        att_linear = self.att_linear1(norm2)
        att_linear = F.leaky_relu(att_linear, self.leaky)
        att_linear = F.dropout(att_linear, 0.5)

        att_linear = self.att_linear2(att_linear)
        att_linear = F.leaky_relu(att_linear, self.leaky)
        att_linear = F.dropout(att_linear, 0.5)

        add3 = torch.add(att_linear, norm2)
        add3 = F.leaky_relu(add3, self.leaky)
        norm3 = self.layer_norm3(add3)

        return norm3


class Decoder(nn.Module):
    def __init__(self, dec_units):
        super(Decoder, self).__init__()
        self.dec_units = dec_units
        self.leaky = 0.2
        self.layers = 1

        self.attention1 = AttentionModule()
        self.attention2 = AttentionModule()

        self.cnn1 = nn.Conv1d(kernel_size=11, in_channels=6, out_channels=6)
        self.cnn2 = nn.Conv1d(kernel_size=11, in_channels=6, out_channels=12)

        self.max1 = nn.MaxPool1d(kernel_size=2)

        self.cnn3 = nn.Conv1d(kernel_size=5, in_channels=12, out_channels=12)
        self.cnn4 = nn.Conv1d(kernel_size=5, in_channels=12, out_channels=24)

        self.max2 = nn.MaxPool1d(kernel_size=2)

        self.cnn5 = nn.Conv1d(kernel_size=3, in_channels=24, out_channels=24)
        self.cnn6 = nn.Conv1d(kernel_size=3, in_channels=24, out_channels=24)

        self.max3 = nn.MaxPool1d(kernel_size=2)

        self.gru = nn.GRU(input_size=24, hidden_size=dec_units, num_layers=self.layers)

        cnn1 = (dec_units - 20) / 2
        cnn2 = (cnn1 - 8) / 2
        cnn3 = (cnn2 - 4) / 2

        self.linear1 = nn.Linear(int(cnn3) * 24, 400)
        self.batch3 = nn.BatchNorm1d(int(cnn3) * dec_units)
        self.linear2 = nn.Linear(400, 400)
        self.linear3 = nn.Linear(404, 100)
        self.batch4 = nn.BatchNorm1d(100)
        self.linear4 = nn.Linear(100, 50)
        self.linear5 = nn.Linear(50, 4)

    def forward(self, last_input, hidden, enc_output):
        att = self.attention1(enc_output)
        att = self.attention2(att)

        context = att.permute(1, 2, 0)

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
        x, gru_hidden = self.gru(context, hidden)
        x = torch.flatten(context)

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
        return x, gru_hidden

    def initialize_hidden(self, window):
        return torch.zeros((self.layers, window, self.dec_units))