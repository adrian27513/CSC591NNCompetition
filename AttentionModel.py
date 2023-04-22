import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
from torch import nn

device = torch.device("cuda:0")


class AttentionModuleEncode(nn.Module):
    def __init__(self):
        super(AttentionModuleEncode, self).__init__()
        self.leaky = 0.2

        self.attention1 = nn.MultiheadAttention(6, 2)
        self.layer_norm1 = nn.LayerNorm(6)

        self.att_linear1 = nn.Linear(6, 6)
        self.att_linear2 = nn.Linear(6, 6)
        self.layer_norm2 = nn.LayerNorm(6)

    def forward(self, enc_output):
        att1, _ = self.attention1(query=enc_output, key=enc_output, value=enc_output)
        att1 = F.dropout(att1, 0.5)
        add1 = torch.add(att1, enc_output)
        add1 = F.leaky_relu(add1, self.leaky)
        norm1 = self.layer_norm1(add1)

        att_linear = self.att_linear1(norm1)
        att_linear = F.leaky_relu(att_linear, self.leaky)
        att_linear = F.dropout(att_linear, 0.5)

        att_linear = self.att_linear2(att_linear)
        att_linear = F.leaky_relu(att_linear, self.leaky)
        att_linear = F.dropout(att_linear, 0.5)

        add2 = torch.add(att_linear, norm1)
        add2 = F.leaky_relu(add2, self.leaky)
        norm2 = self.layer_norm2(add2)

        return norm2


class AttentionModuleDecode(nn.Module):
    def __init__(self):
        super(AttentionModuleDecode, self).__init__()
        self.leaky = 0.2

        self.attention1 = nn.MultiheadAttention(embed_dim=4, num_heads=2)
        self.layer_norm1 = nn.LayerNorm(4)

        self.attention2 = nn.MultiheadAttention(embed_dim=4, num_heads=2, kdim=6, vdim=6)
        self.layer_norm2 = nn.LayerNorm(4)

        self.att_linear1 = nn.Linear(4, 4)
        self.att_linear2 = nn.Linear(4, 4)
        self.layer_norm3 = nn.LayerNorm(4)

    def forward(self, enc_output, dec_output):
        att1, _ = self.attention1(query=dec_output, key=dec_output, value=dec_output)
        att1 = F.dropout(att1, 0.5)
        add1 = torch.add(att1, dec_output)
        add1 = F.leaky_relu(add1, self.leaky)
        norm1 = self.layer_norm1(add1)

        att2, _ = self.attention2(query=norm1, key=enc_output, value=enc_output)
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


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.attention1 = AttentionModuleEncode()
        self.attention2 = AttentionModuleEncode()
        self.attention3 = AttentionModuleEncode()

    def forward(self, x):
        x = self.attention1(x)
        x = self.attention2(x)
        x = self.attention3(x)
        return x


class Decoder(nn.Module):
    def __init__(self, dec_units):
        super(Decoder, self).__init__()
        self.dec_units = dec_units
        self.leaky = 0.2
        self.layers = 3

        self.attention1 = AttentionModuleDecode()
        self.attention2 = AttentionModuleDecode()
        self.attention3 = AttentionModuleDecode()

        self.cnn1 = nn.Conv1d(kernel_size=11, in_channels=6, out_channels=6)
        self.cnn2 = nn.Conv1d(kernel_size=11, in_channels=6, out_channels=12)

        self.max1 = nn.MaxPool1d(kernel_size=2)

        self.cnn3 = nn.Conv1d(kernel_size=5, in_channels=12, out_channels=12)
        self.cnn4 = nn.Conv1d(kernel_size=5, in_channels=12, out_channels=24)

        self.max2 = nn.MaxPool1d(kernel_size=2)

        self.cnn5 = nn.Conv1d(kernel_size=3, in_channels=24, out_channels=24)
        self.cnn6 = nn.Conv1d(kernel_size=3, in_channels=24, out_channels=24)

        self.max3 = nn.MaxPool1d(kernel_size=2)

        self.gru = nn.GRU(input_size=4, hidden_size=dec_units, num_layers=self.layers, dropout=0.3, batch_first=False)

        cnn1 = (dec_units - 20) / 2
        cnn2 = (cnn1 - 8) / 2
        cnn3 = (cnn2 - 4) / 2

        self.linear1a = nn.Linear((int(cnn3) * 24), dec_units)
        self.linear1b = nn.Linear(dec_units, 1)

        self.layer = nn.LayerNorm(dec_units)

        self.linear3 = nn.Linear(dec_units, 100)
        self.linear4 = nn.Linear(100, 50)
        self.linear5 = nn.Linear(50, 4)

    def forward(self, hidden, enc_output, dec_outputs):
        att = self.attention1(enc_output, dec_outputs)
        att = self.attention2(enc_output, att)
        att = self.attention3(enc_output, att)

        context = enc_output.permute(1, 2, 0)
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
        context = torch.flatten(context)

        x, gru_hidden = self.gru(att, hidden)

        xcnn = self.linear1a(context)
        xcnn = F.leaky_relu(xcnn, self.leaky)
        xcnn = F.dropout(xcnn, 0.4)

        xatt = self.linear1b(x)
        xatt = F.leaky_relu(xatt, self.leaky)
        xatt = F.dropout(xatt, 0.4)
        xatt = xatt.squeeze()

        x = torch.add(xatt, xcnn)
        x = self.layer(x)

        x = self.linear3(x)
        x = F.leaky_relu(x, self.leaky)
        x = F.dropout(x, 0.4)
        x = self.linear4(x)
        x = F.leaky_relu(x, self.leaky)
        x = F.dropout(x, 0.4)
        x = self.linear5(x)
        return x, gru_hidden

    def initialize_hidden(self, window):
        return torch.zeros((self.layers, window, self.dec_units))