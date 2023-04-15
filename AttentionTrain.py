import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
import random
import time
from AttentionModel import Encoder, Decoder
from util import get_data
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

# torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda:0")
# device = torch.device("cpu")

window = 1
dec_units = 100
lr = 0.001
# lr = 0.001
mini_batch_size = pow(2, 12)
# mini_batch_size = 1024
training_data, training_data_labels, testing_data, testing_data_labels, loss_weights = get_data(window=window, verbose=True)

encoder = Encoder(output_size=dec_units).to(device)
decoder = Decoder(dec_units=dec_units, output=4).to(device)
encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=lr, betas=(0.9, 0.99))
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr, betas=(0.9, 0.99))

criterion = nn.CrossEntropyLoss()

def train(subject_data):
    total_loss = 0

    encoder.zero_grad()
    decoder.zero_grad()

    input_data = subject_data[0].to(device=device)
    input_label = subject_data[1].to(device=device)
    batch_input = torch.split(input_data, mini_batch_size)
    batch_label = torch.split(input_label, mini_batch_size)

    for batch_in, batch_label in zip(batch_input, batch_label):
        batch_train_in = batch_in.to(device=device)
        batch_train_label = batch_label.to(device=device)
        batch_train_in = F.pad(batch_train_in, (0, 0, 0, 0, 0, mini_batch_size - batch_train_in.shape[0]))
        # batch_train_label = F.pad(batch_train_label, (0, mini_batch_size - batch_train_label.shape[0]))
        batch_train_in = batch_train_in.permute(1,2,0)
        enc_output = encoder(batch_train_in)

        start_value = torch.tensor([-1]).to(device)
        hidden = decoder.initialize_hidden().to(device=device)

        outputs = []
        for i in range(batch_train_label.shape[0]):
            output, hidden = decoder(start_value, hidden, enc_output)
            outputs.append(output)

        output_tensor = torch.cat(outputs)

        loss = criterion(output_tensor, batch_train_label)
        total_loss += loss.item()

        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

    return total_loss


n_iters = 100
plot_every = 10
current_loss = 0
all_losses = []
all_iterCt = []
best_loss = 9999999999999999
print("Start Training")
for epoch in range(1, n_iters + 1):
    encoder.train()
    decoder.train()
    start = time.time()
    subject_list = list(zip(training_data, training_data_labels))
    i = 1
    for batch in subject_list:
        print("Subject: ", i, "/", len(subject_list))
        i += 1
        loss = train(batch)
        current_loss += loss
    print("Epoch", epoch, "| Current Loss:", current_loss)
    if epoch % 1 == 0:
        with torch.no_grad():
            encoder.eval()
            test_losses = []
            total_loss = 0
            pred_sum = 0
            total = 0
            actual = []
            predicted = []
            for subject_data in list(zip(testing_data, testing_data_labels)):
                loss = 0

                input_data = subject_data[0].to(device=device)
                test_label = subject_data[1].to(device=device)
                batch_input = torch.split(input_data, mini_batch_size)
                batch_label = torch.split(test_label, mini_batch_size)

                for batch_in, batch_label in zip(batch_input, batch_label):
                    batch_train_in = batch_in.to(device=device)
                    batch_train_label = batch_label.to(device=device)
                    batch_train_in = F.pad(batch_train_in, (0, 0, 0, 0, 0, mini_batch_size - batch_train_in.shape[0]))
                    # batch_train_label = F.pad(batch_train_label, (0, mini_batch_size - batch_train_label.shape[0]))
                    batch_train_in = batch_train_in.permute(1, 2, 0)
                    enc_output = encoder(batch_train_in)

                    start_value = torch.tensor([-1]).to(device)
                    hidden = decoder.initialize_hidden().to(device=device)

                    outputs = []
                    for i in range(batch_train_label.shape[0]):
                        output, hidden = decoder(start_value, hidden, enc_output)
                        outputs.append(output)

                    output_tensor = torch.cat(outputs)

                    loss = criterion(output_tensor, batch_train_label)

                    out = torch.argmax(output_tensor, dim=-1)
                    predicted.extend(out.cpu())
                    actual.extend(batch_train_label.cpu())
                    equal = torch.eq(out, batch_train_label)
                    pred_sum += torch.sum(equal).item()
                    total += equal.shape[0]
                    test_losses.append(loss.item())
            test_loss = np.average(test_losses)
            confusion = confusion_matrix(actual, predicted)
            f1 = f1_score(actual, predicted, average='macro')
            print(confusion)
            print("Test Loss:", test_loss)
            print("Test Accuracy:", pred_sum/total)
            print("Test F1 Macro", f1)
            if test_loss < best_loss:
                torch.save(encoder.state_dict(),
                           "models/BestAttEncoder_window" + str(window) + "_decunits" + str(dec_units) + ".pt")
                torch.save(decoder.state_dict(),
                           "models/BestAttDecoder_window" + str(window) + "_decunits" + str(dec_units) + ".pt")
                best_loss = test_loss

    # Add current loss avg to list of losses
    if epoch % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        all_iterCt.append(epoch)
        current_loss = 0

    torch.save(encoder.state_dict(), "models/AttEncoder_window"+str(window)+"_decunits"+str(dec_units)+".pt")
    torch.save(decoder.state_dict(), "models/AttDecoder_window"+str(window)+"_decunits"+str(dec_units)+".pt")
    iter_time = time.time() - start
    print("Iter Time:", round(iter_time, 3), "seconds")
    hours_left = iter_time * (n_iters - epoch) / 3600
    hours = int(hours_left)
    minutes = int((hours_left * 60) % 60)
    seconds = int((hours_left * 3600) % 60)
    print("Estimated Time: ", hours, "hours", minutes, "minutes", seconds, "seconds")
    print("=======================")

import matplotlib.pyplot as plt

plt.figure()
plt.plot(all_iterCt, all_losses)
plt.xlabel('Iteration')
plt.ylabel('Training Loss')
plt.savefig("Optimized.png")
# plt.show()
