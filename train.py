import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
import random
import time
from model import RNN
from util import get_data
from sklearn.metrics import confusion_matrix

device = torch.device("cuda:0")
# device = torch.device("cpu")

window = 15
lstm_size = 96
linear_size = 259
dropout = 0.155
training_data, training_data_labels, testing_data, testing_data_labels, loss_weights = get_data(window=window, verbose=True)

rnn = RNN(input_size=6, output_size=4, dropout=dropout, linear_hidden=linear_size, lstm_hidden=lstm_size, lstm_layers=1).to(device)
criterion = nn.CrossEntropyLoss()
# criterion = nn.CrossEntropyLoss()
lr = 0.01
mini_batch_size = 512
optimizer = torch.optim.Adam(rnn.parameters(), lr=lr, betas=(0.9, 0.99))
# optimizer = torch.optim.SGD(rnn.parameters(), lr=lr)
# torch.autograd.set_detect_anomaly(True)


def train(subject_data):
    total_loss = 0
    # input_data = subject_data[0].to(device)
    # input_label = subject_data[1].to(device)
    # output = rnn(input_data)
    # rnn.zero_grad()
    # loss = criterion(output, input_label)
    # loss.backward()
    # optimizer.step()
    input_data = subject_data[0]
    input_label = subject_data[1]
    # print(input_data.shape)
    batch_input = torch.split(input_data, mini_batch_size)
    batch_label = torch.split(input_label, mini_batch_size)
    for batch_in, batch_label in zip(batch_input, batch_label):
        test_in = batch_in.to(device=device)
        test_label = batch_label.to(device=device)
        output = rnn(test_in)
        rnn.zero_grad()
        loss = criterion(output, test_label)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    return output, total_loss


n_iters = 1000
plot_every = 10
current_loss = 0
all_losses = []
all_iterCt = []
best_loss = 9999999999999999
print("Start Training")
for epoch in range(1, n_iters + 1):
    rnn.train()
    start = time.time()
    subject_list = list(zip(training_data, training_data_labels))
    for batch in subject_list:
        output, loss = train(batch)
        current_loss += loss
    print("Epoch", epoch, "| Current Loss:", current_loss)

    with torch.no_grad():
        rnn.eval()
        test_losses = []
        total_loss = 0
        pred_sum = 0
        total = 0
        actual = []
        predicted = []
        for subject_data in list(zip(testing_data, testing_data_labels)):
            input_data = subject_data[0].to(device=device)
            test_label = subject_data[1].to(device=device)
            batch_input = torch.split(input_data, mini_batch_size)
            batch_label = torch.split(test_label, mini_batch_size)
            for batch_in, batch_label in zip(batch_input, batch_label):
                batch_test_in = batch_in.to(device=device)
                batch_test_label = batch_label.to(device=device)
                output = rnn(batch_test_in)
                loss = criterion(output, batch_test_label)
                total_loss += loss.item()
                max_pred = torch.argmax(output, dim=1)
                actual.extend(batch_test_label.tolist())
                predicted.extend(max_pred.tolist())
                equal = torch.eq(batch_test_label, max_pred)
                pred_sum += torch.sum(equal).item()
                total += equal.shape[0]
            test_losses.append(total_loss)
        test_loss = np.average(test_losses)
        confusion = confusion_matrix(actual, predicted)
        print(confusion)
        print("Test Loss:", test_loss)
        print("Test Accuracy:", pred_sum/total)
        if test_loss < best_loss:
            torch.save(rnn.state_dict(), "best_rnn_"+str(window)+"_"+str(int(dropout*10))+"_"+str(linear_size)+"_"+str(lstm_size)+"_"+str(mini_batch_size)+".pt")
            best_loss = test_loss

    # Add current loss avg to list of losses
    if epoch % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        all_iterCt.append(epoch)
        current_loss = 0

    torch.save(rnn.state_dict(), "best_rnn"+str(window)+"_"+str(int(dropout*10))+"_"+str(linear_size)+"_"+str(lstm_size)+".pt")
    iter_time = time.time() - start
    print("Iter Time:", round(iter_time, 3), "seconds")
    hours_left = iter_time * (n_iters - epoch) / 3600
    hours = int(hours_left)
    minutes = int((hours_left * 60) % 60)
    seconds = int((hours_left * 3600) % 60)
    print("Estimated Time: ", hours, "hours", minutes, "minutes", seconds, "seconds")
    print("=======================")

torch.save(rnn.state_dict(), "rnn.pt")

import matplotlib.pyplot as plt

plt.figure()
plt.plot(all_iterCt, all_losses)
plt.xlabel('Iteration')
plt.ylabel('Training Loss')
plt.savefig("Optimized.png")
# plt.show()
