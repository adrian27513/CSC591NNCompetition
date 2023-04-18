import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
import random
import time
from AttentionModel import Decoder
from util import get_data
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

# torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda:0")
# device = torch.device("cpu")

window = 1
mini_batch_size = pow(2, 9)
dec_units = mini_batch_size
teacher_force_ratio = 0.5

training_data, training_data_labels, testing_data, testing_data_labels, loss_weights = get_data(window=window,
                                                                                                verbose=True)

# encoder = Encoder(output_size=dec_units).to(device)
decoder = Decoder(dec_units=dec_units).to(device)
# encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=lr, betas=(0.9, 0.99))
# decoder_optimizer = torch.optim.Adam(decoder.parameters(), betas=(0.9, 0.99), weight_decay=0.01)
decoder_optimizer = torch.optim.SGD(decoder.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.01)
# encoder_scheduler = torch.optim.lr_scheduler.CyclicLR(encoder_optimizer, base_lr=0.0001, max_lr=0.01, step_size_up=500, cycle_momentum=False)
decoder_scheduler = torch.optim.lr_scheduler.CyclicLR(decoder_optimizer, base_lr=0.0001, max_lr=0.01, base_momentum=0.8,
                                                      max_momentum=0.9, step_size_up=1500)
criterion = nn.CrossEntropyLoss()

trainable = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
print("Trainable Parameters:", trainable)


def train(subject_data):
    total_loss = 0

    decoder.zero_grad()

    input_data = subject_data[0].to(device=device)
    input_label = subject_data[1].to(device=device)
    batch_input = torch.split(input_data, mini_batch_size)
    batch_label = torch.split(input_label, mini_batch_size)

    actual = []
    predicted = []
    total_f1 = 0
    batch_len = len(batch_input)
    for batch_in, batch_label in zip(batch_input, batch_label):
        batch_train_in = batch_in
        batch_train_label = batch_label.to(device=device)
        batch_train_in = F.pad(batch_train_in, (0, 0, 0, 0, 0, mini_batch_size - batch_train_in.shape[0])).to(
            device=device)
        # batch_train_label = F.pad(batch_train_label, (0, mini_batch_size - batch_train_label.shape[0]))

        # encoder_hidden = encoder.initialize_hidden()
        #
        # for i in range(batch_train_in.shape[0]):
        #     train_input = batch_train_in[0]
        #     encoder_hidden = encoder(train_input, encoder_hidden)
        #
        last_output = torch.tensor([0, 0, 0, 0]).to(device=device)
        decoder_hidden = decoder.initialize_hidden(window).to(device=device)

        use_teacher_forcing = True if random.random() < teacher_force_ratio else False

        outputs = []
        for i in range(batch_train_label.shape[0]):
            last_output, hidden = decoder(last_output, decoder_hidden, batch_train_in)
            outputs.append(last_output.unsqueeze(0))
            if use_teacher_forcing:
                last_output = last_output * 0
                last_output[batch_train_label[i].item()] = 1
            else:
                last_output = F.softmax(last_output, dim=-1)

        output_tensor = torch.cat(outputs, dim=0)

        out = torch.argmax(output_tensor, dim=-1)
        predicted.extend(out.cpu())
        actual.extend(batch_train_label.cpu())
        f1 = f1_score(actual, predicted, average='macro')
        total_f1 += f1
        loss = criterion(output_tensor, batch_train_label)
        total_loss += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=5)

        decoder_optimizer.step()
        decoder_scheduler.step()

    return total_loss, total_f1 / batch_len


n_iters = 100
plot_every = 1
current_loss = 0
current_test_loss = 0

current_train_f1 = 0
current_test_f1 = 0
all_losses = []
all_test_losses = []
all_iterCt = []

all_f1_train = []
all_f1_test = []
best_loss = 9999999999999999
best_f1 = 0
print("Start Training")
for epoch in range(1, n_iters + 1):
    decoder.train()
    start = time.time()
    subject_list = list(zip(training_data, training_data_labels))
    i = 1
    current_f1 = 0
    for batch in subject_list:
        print("Subject: ", i, "/", len(subject_list))
        i += 1
        loss, f1 = train(batch)
        print(loss)
        current_loss += loss
        current_f1 += f1
    current_f1 /= len(subject_list)
    current_train_f1 += current_f1
    print("Epoch", epoch, "| Current Loss:", current_loss, "| Current F1:", current_f1)
    if epoch % 1 == 0:
        with torch.no_grad():
            decoder.eval()
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

                    last_output = torch.tensor([0, 0, 0, 0]).to(device=device)
                    decoder_hidden = decoder.initialize_hidden(window).to(device=device)

                    outputs = []
                    for i in range(batch_train_label.shape[0]):
                        last_output, hidden = decoder(last_output, decoder_hidden, batch_train_in)
                        outputs.append(last_output.unsqueeze(0))
                        last_output = F.softmax(last_output, dim=-1)

                    output_tensor = torch.cat(outputs, dim=0)

                    loss = criterion(output_tensor, batch_train_label)

                    out = torch.argmax(output_tensor, dim=-1)
                    predicted.extend(out.cpu())
                    actual.extend(batch_train_label.cpu())
                    equal = torch.eq(out, batch_train_label)
                    pred_sum += torch.sum(equal).item()
                    total += equal.shape[0]
                    test_losses.append(loss.item())
            test_loss = np.average(test_losses)
            current_test_loss += test_loss
            confusion = confusion_matrix(actual, predicted)
            f1 = f1_score(actual, predicted, average='macro')
            current_test_f1 += f1
            print(confusion)
            print("Test Loss:", test_loss)
            print("Test Accuracy:", pred_sum / total)
            print("Test F1 Macro", f1)
            if test_loss < best_loss or f1 > best_f1:
                # torch.save(encoder.state_dict(),
                #            "models/BestAttEncoder_window" + str(window) + "_decunits" + str(dec_units) + ".pt")
                torch.save(decoder.state_dict(),
                           "models/BestAttDecoder_window" + str(window) + "_decunits" + str(dec_units) + ".pt")
                best_loss = test_loss
                best_f1 = f1

    # Add current loss avg to list of losses
    if epoch % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        all_test_losses.append(current_test_loss / plot_every)
        all_f1_train.append(current_train_f1 / plot_every)
        all_f1_test.append(current_test_f1 / plot_every)

        all_iterCt.append(epoch)
        plt.figure()
        plt.plot(all_iterCt, all_losses, color="blue")
        plt.xlabel('Iteration')
        plt.ylabel('Training Loss')
        plt.savefig("models/Loss.png")
        plt.clf()

        plt.figure()
        plt.plot(all_iterCt, all_test_losses, color="red")
        plt.xlabel('Iteration')
        plt.ylabel('Test Loss')
        plt.savefig("models/TestLoss.png")
        plt.clf()

        plt.figure()
        plt.plot(all_iterCt, all_f1_train, color="red")
        plt.plot(all_iterCt, all_f1_test, color="blue")
        plt.xlabel('Iteration')
        plt.ylabel('F1')
        plt.savefig("models/F1.png")
        plt.clf()

        current_loss = 0
        current_test_loss = 0
        current_train_f1 = 0
        current_test_f1

    # torch.save(encoder.state_dict(), "models/AttEncoder_window"+str(window)+"_decunits"+str(dec_units)+".pt")
    torch.save(decoder.state_dict(), "models/AttDecoder_window" + str(window) + "_decunits" + str(dec_units) + ".pt")
    iter_time = time.time() - start
    print("Iter Time:", round(iter_time, 3), "seconds")
    hours_left = iter_time * (n_iters - epoch) / 3600
    hours = int(hours_left)
    minutes = int((hours_left * 60) % 60)
    seconds = int((hours_left * 3600) % 60)
    print("Estimated Time: ", hours, "hours", minutes, "minutes", seconds, "seconds")
    print("=======================")

# plt.show()
