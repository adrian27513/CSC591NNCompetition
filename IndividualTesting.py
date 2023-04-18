import pandas as pd
import torch
import torch.nn.functional as F
from model import RNN
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import numpy as np
from AttentionModel import Decoder
from torch import nn

device = torch.device("cuda")

def align_data(subject):
    gyro_names = ['X_acc', 'Y_acc', 'Z_acc', 'X', 'Y', 'Z']
    gyro = pd.read_csv('TrainingData/' + subject + '__x.csv', names=gyro_names)
    gyro = (gyro - gyro.mean()) / (gyro.std())
    gyro['time'] = pd.read_csv('TrainingData/' + subject + '__x_time.csv', header=None)
    gyro['time'] = pd.to_timedelta(gyro['time'], unit='S')

    labels = pd.read_csv('TrainingData/' + subject + '__y.csv', names=['label'])
    time_labels = pd.read_csv('TrainingData/' + subject + '__y_time.csv', names=['time'])
    labels['time'] = time_labels - time_labels['time'][0]
    labels['time'] = pd.to_timedelta(labels['time'], unit='S')
    labels = labels.set_index('time')
    labels = labels.resample('0.025S').asfreq().ffill()
    labels.reset_index(inplace=True)
    labels = labels.rename(columns={'index': 'time'})

    merge = pd.merge(gyro, labels, how='inner', on='time')
    merge = merge.drop(columns=['time'])
    return merge

aligned_pandas = align_data("subject_006_02")
labels_pandas = aligned_pandas['label']
data_pandas = aligned_pandas.drop(columns="label")

input_data = torch.from_numpy(data_pandas.values).to(device).to(torch.float32)
test_label = torch.from_numpy(labels_pandas.values).to(device).to(torch.long)
input_data = input_data.unsqueeze(1)

criterion = nn.CrossEntropyLoss()
window = 1
dec_units = 512
decoder = Decoder(dec_units=dec_units).to(device)
# decoder.load_state_dict(torch.load("models/BestAttDecoder_window1_decunits512.pt"))

mini_batch_size = dec_units
with torch.no_grad():
    decoder.eval()
    test_losses = []
    total_loss = 0
    pred_sum = 0
    total = 0
    actual = []
    predicted = []
    loss = 0

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
            # print(last_output)
            last_output = F.softmax(last_output, dim=-1)

        # print(outputs)
        output_tensor = torch.cat(outputs, dim=0).to(torch.float32)
        # print(output_tensor)
        # print(F.softmax(output_tensor, dim=-1))

        loss = criterion(output_tensor, batch_train_label)

        print(loss)

        out = torch.argmax(output_tensor, dim=-1)
        predicted.extend(out.cpu())
        actual.extend(batch_train_label.cpu())
        equal = torch.eq(out, batch_train_label)
        pred_sum += torch.sum(equal).item()
        total += equal.shape[0]
        test_losses.append(loss.item())
print(test_losses)
test_loss = np.average(test_losses)
confusion = confusion_matrix(actual, predicted)
f1 = f1_score(actual, predicted, average='macro')
print(confusion)
print("Test Loss:", test_loss)
print("Test Accuracy:", pred_sum / total)
print("Test F1 Macro", f1)


