import torch
from model import RNN
import pandas as pd
device = torch.device("cuda")

def align_data(subject):
    gyro_names = ['X_acc', 'Y_acc', 'Z_acc', 'X', 'Y', 'Z']
    gyro = pd.read_csv('TestData/' + subject + '__x.csv', names=gyro_names)
    gyro = (gyro - gyro.mean()) / (gyro.std())
    return gyro

input_data = torch.from_numpy(align_data("subject_009_01").values).to(device)

window = 15
lstm_size = 500
linear_size = 278
dropout = 0.175
rnn = RNN(input_size=6, output_size=4, dropout=dropout, linear_hidden=linear_size, lstm_hidden=lstm_size, lstm_layers=1).to(device)
rnn.load_state_dict(torch.load("models/best_rnn_15_17_278_500_1024.pt"))
rnn.eval()

with torch.no_grad():
    pred_sum = 0
    total = 0
    actual = []
    predicted = []
    state = None
    batch_input = torch.split(input_data, mini_batch_size)
    batch_label = torch.split(test_label, mini_batch_size)

    for batch_in, batch_label in zip(batch_input, batch_label):
        test_in = batch_in.to(device=device)
        test_in = F.pad(test_in, (0, 0, 0, 0, 0, mini_batch_size - test_in.shape[0]))
        test_label = batch_label.to(device=device)
        test_in = F.pad(test_in, (0, 0, 0, 0, 0, mini_batch_size - test_in.shape[0]))
        test_label = F.pad(test_label, (0, mini_batch_size - test_label.shape[0]))
        output, state = rnn(test_in, state)
        max_pred = torch.argmax(output, dim=1)
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