import pandas as pd
import torch
import torch.nn.functional as F
from model import RNN
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

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

aligned_pandas = align_data("subject_006_01")
labels_pandas = aligned_pandas['label']
data_pandas = aligned_pandas.drop(columns="label")

input_data = torch.from_numpy(data_pandas.values).to(device)
test_label = torch.from_numpy(labels_pandas.values).to(device)

window = 15
lstm_size = 500
linear_size = 278
dropout = 0.175
rnn = RNN(input_size=6, output_size=4, dropout=dropout, linear_hidden=linear_size, lstm_hidden=lstm_size, lstm_layers=1).to(device)
rnn.load_state_dict(torch.load("models/best_rnn_15_17_278_500_1024.pt"))
rnn.eval()

mini_batch_size = 1024
with torch.no_grad():
    pred_sum = 0
    total = 0
    actual = []
    predicted = []
    state = None
    batch_input = torch.split(input_data, mini_batch_size)
    batch_label = torch.split(test_label, mini_batch_size)

    for batch_in, batch_label in zip(batch_input, batch_label):
        test_in = batch_in.to(device=device).to(torch.float32)
        test_in = torch.unsqueeze(test_in, dim=1)

        test_label = batch_label.to(device=device).to(torch.float32)

        test_in = F.pad(test_in, (0, 0, 0, 0, 0, mini_batch_size - test_in.shape[0]))
        test_label = F.pad(test_label, (0, mini_batch_size - test_label.shape[0]))

        # test_label = torch.unsqueeze(test_label, dim=1)

        output, state = rnn(test_in, state)
        max_pred = torch.argmax(output, dim=1)
        print(max_pred.shape)
        print(test_label.shape)
        equal = torch.eq(test_label, max_pred)
        pred_sum += torch.sum(equal).item()
        total += equal.shape[0]

        gt_list = test_label.tolist()
        pred_list = max_pred.tolist()
        # X = list(range(len(gt_list)))
        actual.extend(gt_list)
        predicted.extend(pred_list)
        # plt.plot(X, gt_list, label='gt')
        # plt.plot(X, pred_list, label='pred', alpha=0.3)
        # plt.legend(loc="upper left")
        # plt.show()
        # plt.clf()
    f1 = f1_score(actual, predicted, average='macro')
    confusion = confusion_matrix(actual, predicted)
    print(confusion)
    print("F1:", f1)
    print("Accuracy:",pred_sum / total)