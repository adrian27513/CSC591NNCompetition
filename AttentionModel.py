import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from torch import nn
import torch
import torch.nn.functional as F

device = torch.device("cuda:0")

class Encoder(nn.Module):
    def __init__(self, output_size):
        super(Encoder, self).__init__()
        self.leaky = 0.1
        self.cnn1 = nn.Conv1d(kernel_size=10, in_channels=6, out_channels=6)
        self.cnn2 = nn.Conv1d(kernel_size=10, in_channels=6, out_channels=6)

        self.max1 = nn.MaxPool1d(kernel_size=3)

        self.cnn3 = nn.Conv1d(kernel_size=5, in_channels=6, out_channels=12)
        self.cnn4 = nn.Conv1d(kernel_size=5, in_channels=12, out_channels=12)

        self.max2 = nn.MaxPool1d(kernel_size=2)

        self.gru = nn.GRU(input_size=12, hidden_size=100, batch_first=True)

        self.linear1 = nn.Linear(100, 100)
        self.linear2 = nn.Linear(100, 256)
        self.linear3 = nn.Linear(256, 100)
        self.linear4 = nn.Linear(100, output_size)


    def forward(self, x):
        x = self.cnn1(x)
        x = F.leaky_relu(x, self.leaky)
        x = self.cnn2(x)
        x = F.leaky_relu(x, self.leaky)
        x = self.max1(x)
        x = self.cnn3(x)
        x = F.leaky_relu(x, self.leaky)
        x = self.cnn4(x)
        x = F.leaky_relu(x, self.leaky)
        x = self.max2(x)
        x = x.permute(0,2,1)
        x, output = self.gru(x)
        x = F.leaky_relu(x, self.leaky)
        x = F.dropout(x, 0.5)

        x = self.linear1(x)
        x = F.leaky_relu(x, self.leaky)
        x = F.dropout(x, 0.5)

        x = self.linear2(x)
        x = F.leaky_relu(x, self.leaky)
        x = F.dropout(x, 0.5)

        x = self.linear3(x)
        x = F.leaky_relu(x, self.leaky)
        x = F.dropout(x, 0.5)

        x = self.linear4(x)
        x = torch.squeeze(x)
        return x

    def initialize_hidden(self, batch):
        return torch.zeros((1, batch, self.dec_units))

class Decoder(nn.Module):
    def __init__(self, dec_units, output):
        super(Decoder, self).__init__()
        self.dec_units = dec_units
        self.leaky = 0.1

        self.W1 = nn.Linear(dec_units, dec_units)
        self.W2 = nn.Linear(4, dec_units)
        self.V = nn.Linear(2*dec_units, dec_units)

        self.combine = nn.Linear(167, 500)

        self.gru = nn.GRU(input_size=500, hidden_size=dec_units)

        self.linear1 = nn.Linear(dec_units, 400)
        self.linear2 = nn.Linear(400, 100)
        self.linear3 = nn.Linear(100, output)

    def forward(self, x, hidden, enc_output):
        embed = [0,0,0,0]
        embed[x.item()] = 1
        embed_tensor = torch.tensor(embed, device=device, dtype=torch.float32)
        w1 = self.W1(hidden)
        w1 = F.leaky_relu(w1, self.leaky).squeeze(0)
        w2 = self.W2(embed_tensor)
        w2 = F.leaky_relu(w2, self.leaky)

        cat = torch.cat((w1, w2))
        score = self.V(cat)
        score = F.leaky_relu(score, self.leaky)

        weights = F.softmax(score, dim=-1)
        context = torch.matmul(enc_output, weights).squeeze()
        context = torch.concat((context, embed_tensor), dim=0).unsqueeze(0)
        x = self.combine(context)
        x = F.leaky_relu(x, self.leaky)
        x, hidden = self.gru(x, hidden)
        x = F.leaky_relu(x, self.leaky)

        x = self.linear1(x)
        x = F.leaky_relu(x, self.leaky)
        x = F.dropout(x, 0.5)

        x = self.linear2(x)
        x = F.leaky_relu(x, self.leaky)
        x = F.dropout(x, 0.5)

        x = self.linear3(x)

        return x, hidden

    def initialize_hidden(self):
        return torch.zeros((1, self.dec_units))