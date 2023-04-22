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
from util import get_balanced_data, GyroDataset
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
import random

# torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.benchmark = True
device = torch.device("cuda:0")
# device = torch.device("cpu")

window = 1
mini_batch_size = pow(2, 8)
dec_units = mini_batch_size
teacher_force_ratio = 0.5

training_data, training_data_labels, testing_data, testing_data_labels = get_balanced_data(window, True)

decoder = Decoder(dec_units=dec_units).to(device)
# decoder = torch.compile(decoder)
encoder = Encoder().to(device)
# encoder = torch.compile(encoder)

# decoder.load_state_dict(torch.load("models/AttDecoder_window1_decunits512.pt"))
encoder_optimizer = torch.optim.AdamW(encoder.parameters(), lr=0.001)
decoder_optimizer = torch.optim.AdamW(decoder.parameters(), lr=0.001)
decoder_scheduler = torch.optim.lr_scheduler.CyclicLR(decoder_optimizer, base_lr=0.001, max_lr=0.01, step_size_up=830,
                                                      cycle_momentum=False, mode='triangular2')
encoder_scheduler = torch.optim.lr_scheduler.CyclicLR(encoder_optimizer, base_lr=0.001, max_lr=0.01, step_size_up=830,
                                                      cycle_momentum=False, mode='triangular2')
# decoder_optimizer = torch.optim.SGD(decoder.parameters(), lr=0.01, momentum=0.9)
# decoder_scheduler = torch.optim.lr_scheduler.CyclicLR(decoder_optimizer, base_lr=0.001, max_lr=0.01, base_momentum=0.8, max_momentum=0.9,  step_size_up=100)
criterion = nn.CrossEntropyLoss()
scaler = torch.cuda.amp.GradScaler()

trainable = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
print("Trainable Parameters:", trainable)


def forward(data, labels, train):
    last_output = torch.tensor([0, 0, 0, 0]).to(device=device)
    decoder_hidden = decoder.initialize_hidden(window).to(device=device)

    if train:
        use_teacher_forcing = True if random.random() < teacher_force_ratio else False
    else:
        use_teacher_forcing = False
    enc_out = encoder(data)
    outputs = torch.zeros((labels.shape[0], 1, 4), device=device).detach()
    pred_out = []
    for i in range(labels.shape[0]):
        last_output, decoder_hidden = decoder(decoder_hidden, enc_out, outputs)
        # print(last_output)
        pred_out.append(last_output)
        if use_teacher_forcing:
            last_output = last_output * 0
            last_output[labels[i].item()] = 1
        else:
            last_output = F.softmax(last_output, dim=-1)
        outputs[i][0] = last_output
    output_tensor = torch.stack(pred_out)
    return output_tensor


def train(subject_data):
    encoder.zero_grad(set_to_none=True)
    decoder.zero_grad(set_to_none=True)

    data = subject_data[0].to(device=device)
    labels = subject_data[1].to(device=device)

    with torch.autocast(device_type='cuda', dtype=torch.float16):
        output_tensor = forward(data, labels, True)
        out = torch.argmax(output_tensor, dim=-1)
        f1 = f1_score(out.cpu(), labels.cpu(), average='macro')
        loss = criterion(output_tensor, labels)

    scaler.scale(loss).backward()
    # loss.backward()

    scaler.unscale_(decoder_optimizer)
    scaler.unscale_(encoder_optimizer)

    torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=5)
    torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=5)
    scaler.step(decoder_optimizer)
    scaler.step(encoder_optimizer)
    scaler.update()

    #    confusion = confusion_matrix(actual, predicted)
    #    print(confusion)
    #    print("F1:", f1)
    #    print("Loss:", loss.item())
    #    print("---------")
    total_loss = loss.item()

    decoder_scheduler.step()
    encoder_scheduler.step()

    return total_loss, f1


n_iters = 200
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

merged_train = []

subject_list = list(zip(training_data, training_data_labels))
random.shuffle(subject_list)

for subject in subject_list:
    input_data = subject[0]
    input_label = subject[1]
    batch_input = torch.split(input_data, mini_batch_size)
    batch_label = torch.split(input_label, mini_batch_size)
    print("Full Batch Len:", len(batch_input))
    batch_list = list(zip(batch_input, batch_label))
    if len(batch_list) != 1:
        batch_list.pop()

    merged_train.extend(batch_list)

train_data = GyroDataset(merged_train)

train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)

print("Start Training")

for epoch in range(1, n_iters + 1):
    encoder.train()
    decoder.train()
    random.shuffle(merged_train)
    start = time.time()
    i = 1
    for data, label in train_dataloader:
        batch_start = time.time()
        loss, f1 = train((data.squeeze(0), label.squeeze(0)))

        #      print("Batch: ", i, "/", len(merged_train))
        #      print("Batch Loss", loss)
        #      print("Batch F1", f1)
        current_loss += loss
        current_train_f1 += f1
        i += 1
        batch_iter_time = time.time() - batch_start
    #      print("Batch Iter Time:", round(batch_iter_time, 3), "seconds")
    current_loss /= len(merged_train)
    current_train_f1 /= len(merged_train)
    print("Epoch", epoch, "| Current Loss:", current_loss, "| Current F1:", current_train_f1)
    if epoch % 1 == 0:
        with torch.no_grad():
            encoder.eval()
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

                batch_list = list(zip(batch_input, batch_label))
                if len(batch_list) != 1:
                    batch_list.pop()

                random.shuffle(batch_list)

                batch_len = len(batch_list)
                # print("Batch Length:", batch_len)

                random.shuffle(batch_list)
                b_in, b_label = zip(*batch_list)

                for batch_in, batch_label in zip(b_in, b_label):
                    batch_train_in = batch_in.to(device=device)
                    batch_train_label = batch_label.to(device=device)

                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        output_tensor = forward(batch_train_in, batch_train_label, False)

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
                torch.save(encoder.state_dict(),
                           "models/BestAttEncoder_decunits" + str(dec_units) + ".pt")
                torch.save(decoder.state_dict(),
                           "models/BestAttDecoder_decunits" + str(dec_units) + ".pt")
                best_loss = test_loss
                best_f1 = f1
    # Add current loss avg to list of losses
    all_losses.append(current_loss)
    all_test_losses.append(current_test_loss)
    all_f1_train.append(current_train_f1)
    all_f1_test.append(current_test_f1)
    all_iterCt.append(epoch)

    if epoch % 5 == 0:
        plt.figure()
        plt.plot(all_iterCt, all_losses, color="red", label="Train")
        plt.plot(all_iterCt, all_test_losses, color="blue", label="Test")
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend(loc="upper left")
        plt.savefig("models/Loss.png")
        plt.clf()

        plt.plot(all_iterCt, all_f1_train, color="red", label="Train")
        plt.plot(all_iterCt, all_f1_test, color="blue", label="Test")
        plt.xlabel('Iteration')
        plt.ylabel('F1')
        plt.legend(loc="upper left")
        plt.savefig("models/F1.png")
        plt.clf()

        plt.close()
    #    print("=======================")
    #    print("Train Losses:", all_losses)
    #    print("Test Losses:", all_test_losses)
    #    print("Train F1:", all_f1_train)
    #    print("Test F1:", all_f1_test)
    #    print("=======================")
    current_loss = 0
    current_test_loss = 0
    current_train_f1 = 0
    current_test_f1 = 0

    torch.save(encoder.state_dict(), "models/AttEncoder_decunits" + str(dec_units) + ".pt")
    torch.save(decoder.state_dict(), "models/AttDecoder_decunits" + str(dec_units) + ".pt")
    iter_time = time.time() - start
    print("Iter Time:", round(iter_time, 3), "seconds")
    hours_left = iter_time * (n_iters - epoch) / 3600
    hours = int(hours_left)
    minutes = int((hours_left * 60) % 60)
    seconds = int((hours_left * 3600) % 60)
    print("Estimated Time: ", hours, "hours", minutes, "minutes", seconds, "seconds")
    print("=======================")
    print("=-=-=-=-=-=-=-=-=-=-=-=")

# plt.show()