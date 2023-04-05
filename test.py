import torch
from model import RNN
from util import get_data
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

device = torch.device("cuda:0")

training_data, training_data_labels, testing_data, testing_data_labels, _ = get_data(5, True)

rnn = RNN(input_size=6, output_size=4, dropout=0.2, linear_hidden=30, lstm_hidden=30).to(device)
# rnn.load_state_dict(torch.load("models/best_rnn_optimize_1.pt"))
rnn.eval()
with torch.no_grad():
    pred_sum = 0
    total = 0
    actual = []
    predicted = []

    for subject_data in list(zip(testing_data, testing_data_labels)):
        input_data = subject_data[0].to(device)
        test_label = subject_data[1].to(device)
        output = rnn(input_data)
        max_pred = torch.argmax(output, dim=1)
        equal = torch.eq(test_label, max_pred)
        pred_sum += torch.sum(equal).item()
        total += equal.shape[0]

        gt_list = test_label.tolist()
        pred_list = max_pred.tolist()
        print(pred_list)
        X = list(range(len(gt_list)))
        actual.extend(gt_list)
        predicted.extend(pred_list)
        # plt.plot(X, gt_list, label='gt')
        # plt.plot(X, pred_list, label='pred', alpha=0.3)
        # plt.legend(loc="upper left")
        # plt.show()
        # plt.clf()
    confusion = confusion_matrix(actual, predicted)
    print(confusion)
    print(pred_sum / total)