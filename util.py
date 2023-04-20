import pandas as pd
import os
import torch
import numpy as np
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
import pickle

device = torch.device("cuda")


def balance(merge, strict):
    counts = merge['class'].value_counts()
    keys = counts.keys().tolist()
    values = counts.tolist()
    average = 0
    zeros = 0
    ones = 0
    twos = 0
    threes = 0

    try:
        zeros = int(values[keys.index(0.0)])
    except:
        pass
    try:
        ones = int(values[keys.index(1.0)])
    except:
        pass

    try:
        twos = int(values[keys.index(2.0)])
    except:
        pass

    try:
        threes = int(values[keys.index(3.0)])
    except:
        pass

    values = [zeros, ones, twos, threes]
    values.sort()

    if (strict):
        average = min(values)
    else:
        average = 1.3 * (values[0] + values[1]) / 2

    if average != 0:
        merge = merge.reset_index().drop(columns='index')
        zero_df = merge[merge['class'] == 0].sample(n=max(0, zeros - int(average)))
        merge = merge.drop(zero_df.index)

        merge = merge.reset_index().drop(columns='index')
        ones_df = merge[merge['class'] == 1].sample(n=max(0, ones - int(average)))
        merge = merge.drop(ones_df.index)

        merge = merge.reset_index().drop(columns='index')
        twos_df = merge[merge['class'] == 2].sample(n=max(0, twos - int(average)))
        merge = merge.drop(twos_df.index)

        merge = merge.reset_index().drop(columns='index')
        three_df = merge[merge['class'] == 3].sample(n=max(0, threes - int(average)))
        merge = merge.drop(three_df.index)
        merge = merge.reset_index().drop(columns='index')

        print(merge['class'].value_counts())
    return merge


def get_balanced_data(window, verbose):
    balanced = pd.read_pickle('balancedDataset.pkl')

    training_data = []
    training_data_labels = []

    testing_data = []
    testing_data_labels = []
    leave_out = [5, 2, 2, 1, 1, 2, 3, -1]
    for i, subject in enumerate(balanced.keys()):
        if verbose:
            print("Subject", i + 1)
        df = balanced[subject]
        df = df.drop(columns=["time", "class_time"])
        df = balance(df, True)

        train = df[df['session'] != leave_out[i]]
        train = train.drop(columns='session')
        train_data = train.drop(columns='class')
        train_data = (train_data - train_data.mean()) / (train_data.std())
        train_data_label = train['class']

        training = train_data.to_numpy()
        training_labels = train_data_label.to_numpy()
        sub_windows = (-(window - 1) + np.expand_dims(np.arange(window), 0) + np.expand_dims(np.arange(len(training)),
                                                                                             0).T)[window - 1:]
        train_labels = torch.mode(torch.from_numpy(training_labels[sub_windows]), dim=-1)[0].to(torch.long)
        sub_features = torch.from_numpy(training[sub_windows])
        train_labels = torch.mode(torch.from_numpy(training_labels[sub_windows]), dim=-1)[0].to(torch.long)

        training_data.append(sub_features.to(torch.float32))
        training_data_labels.append(train_labels.to(torch.long))

        if subject != "subject_8":
            test = df[df['session'] == leave_out[i]]
            test = test.drop(columns='session')
            test_data = test.drop(columns='class')
            test_data = (test_data - test_data.mean()) / (test_data.std())
            test_data_label = test['class']

            testing = test_data.to_numpy()
            testing_labels = test_data_label.to_numpy()
            sub_windows = (-(window - 1) + np.expand_dims(np.arange(window), 0) + np.expand_dims(
                np.arange(len(testing)),
                0).T)[window - 1:]
            sub_features = torch.from_numpy(testing[sub_windows])
            test_labels = torch.mode(torch.from_numpy(testing_labels[sub_windows]), dim=-1)[0].to(torch.long)

            testing_data.append(sub_features.to(torch.float32))
            testing_data_labels.append(test_labels.to(torch.long))
    if verbose:
        print("Done Getting Data")
        print("=================")
    return training_data, training_data_labels, testing_data, testing_data_labels

# # from imblearn.under_sampling import RandomUnderSampler
# def align_data(subject):
#     gyro_names = ['X_acc', 'Y_acc', 'Z_acc', 'X', 'Y', 'Z']
#     gyro = pd.read_csv('TrainingData/' + subject + '__x.csv', names=gyro_names)
#     gyro = (gyro - gyro.mean()) / (gyro.std())
#     gyro['time'] = pd.read_csv('TrainingData/' + subject + '__x_time.csv', header=None)
#     gyro['time'] = pd.to_timedelta(gyro['time'], unit='S')
#
#     labels = pd.read_csv('TrainingData/' + subject + '__y.csv', names=['label'])
#     time_labels = pd.read_csv('TrainingData/' + subject + '__y_time.csv', names=['time'])
#     labels['time'] = time_labels - time_labels['time'][0]
#     labels['time'] = pd.to_timedelta(labels['time'], unit='S')
#     labels = labels.set_index('time')
#     labels = labels.resample('0.025S').asfreq().ffill()
#     labels.reset_index(inplace=True)
#     labels = labels.rename(columns={'index': 'time'})
#
#     merge = pd.merge(gyro, labels, how='inner', on='time')
#     merge = merge.drop(columns=['time'])
#
#     counts = merge['label'].value_counts()
#     keys = counts.keys().tolist()
#     values = counts.tolist()
#     average = 0
#     zeros = 0
#     ones = 0
#     twos = 0
#     threes = 0
#
#     try:
#         zeros = int(values[keys.index(0.0)])
#     except:
#         pass
#     try:
#         ones = int(values[keys.index(1.0)])
#     except:
#         pass
#
#     try:
#         twos = int(values[keys.index(2.0)])
#     except:
#         pass
#
#     try:
#         threes = int(values[keys.index(3.0)])
#     except:
#         pass
#
#     average = (ones + twos + threes) / 4
#     # average = 10
#
#     # print(merge['label'].value_counts())
#     # print(average)
#     # merge.plot()
#     # plt.show()
#     # plt.clf()
#     # merge['label'].value_counts().plot.bar()
#     # plt.show()
#     # plt.clf()
#     if average != 0:
#         zero_df = merge[merge.label == 0].sample(n=max(0,zeros - int(average*.5)))
#         merge = merge.drop(zero_df.index)
#
#         ones_df = merge[merge.label == 1].sample(n=max(0,ones - int(average/3)))
#         merge = merge.drop(ones_df.index)
#
#         twos_df = merge[merge.label == 2].sample(n=max(0,twos - int(average)))
#         merge = merge.drop(twos_df.index)
#
#         three_df = merge[merge.label == 3].sample(n=max(0,threes - int(average)))
#         merge = merge.drop(three_df.index)
#
#         merge = merge.reset_index().drop(columns='index')
#
#     # merge['label'].value_counts().plot.bar()
#     # plt.show()
#     # print(merge)
#
#     # print(merge['label'].value_counts())
#     # merge.plot()
#     # plt.show()
#     return merge
#
#
# def get_data(window, verbose):
#     # print(os.listdir('TrainingData'))
#     training_list = list(set(e.split('__')[0] for e in os.listdir('TrainingData')))
#     if verbose:
#         print(training_list)
#         print(len(training_list))
#     training_data = []
#     training_data_labels = []
#     testing_data = []
#     testing_data_labels = []
#     split = 0.2
#
#     train_amount = pd.Series([0, 0, 0, 0], index=[0.0, 1.0, 2.0, 3.0])
#
#     for i, data in enumerate(training_list):
#         if "subject_006" in data:
#             if verbose:
#                 print("Skipping Subject 6")
#             continue
#         if verbose:
#             print("Subject", i, ":", data)
#         aligned = align_data(data)
#         df_data = aligned.drop(columns=['label'])
#         df_label = aligned['label']
#
#         train_amount = train_amount.add(df_label.iloc[:int(len(aligned.index) * split)].value_counts(), fill_value=0)
#         test_split = int(len(aligned.index) * split)
#         # test_start = random.randint(0, len(aligned.index) - test_split)
#         test_start = len(aligned.index) - test_split
#
#         # Train
#         training = df_data.drop(df_data.index[test_start:test_start + test_split]).to_numpy()
#         training_labels = df_label.drop(df_data.index[test_start:test_start + test_split]).to_numpy()
#         # training_labels = (training_labels * 0) + 2
#         sub_windows = (-(window - 1) + np.expand_dims(np.arange(window), 0) + np.expand_dims(np.arange(len(training)),
#                                                                                              0).T)[window - 1:]
#         sub_features = torch.from_numpy(training[sub_windows])
#         train_labels = torch.mode(torch.from_numpy(training_labels[sub_windows]), dim=-1)[0].to(torch.long)
#         # padded = []
#         # for idx in range(1, window):
#         #     feature = torch.from_numpy(training[max(0,idx-window):idx])
#         #     length = feature.shape[0]
#         #     feature_padded = F.pad(feature, (0,0,window - length,0), value=-1)
#         #     padded.append(feature_padded)
#         # if window != 1:
#         #     stacked_padded = torch.stack(padded)
#         #     train_features = torch.cat((stacked_padded, sub_features), 0).to(torch.float32)
#         # else:
#
#         train_features = sub_features.to(torch.float32)
#         training_data.append(train_features)
#         training_data_labels.append(train_labels)
#
#         # Test
#         testing = df_data.iloc[test_start:test_start + test_split, :].to_numpy()
#         testing_labels = df_label.to_numpy()[test_start:test_start + test_split]
#         sub_windows = (-(window - 1) + np.expand_dims(np.arange(window), 0) + np.expand_dims(np.arange(len(testing)),
#                                                                                              0).T)[window - 1:]
#         sub_features = torch.from_numpy(testing[sub_windows])
#         test_labels = torch.mode(torch.from_numpy(testing_labels[sub_windows]), dim=-1)[0].to(torch.long)
#
#         # padded = []
#         # for idx in range(1, window):
#         #     feature = torch.from_numpy(testing[max(0, idx - window):idx])
#         #     length = feature.shape[0]
#         #     feature_padded = F.pad(feature, (0, 0, window - length, 0), value=0)
#         #     padded.append(feature_padded)
#         # if window != 1:
#         #     stacked_padded = torch.stack(padded)
#         #     test_features = torch.cat((stacked_padded, sub_features), 0).to(torch.float32)
#         # else:
#         test_features = sub_features.to(torch.float32)
#         test_labels = torch.tensor(testing_labels).to(torch.long)
#         testing_data.append(test_features)
#         testing_data_labels.append(test_labels)
#
#     train_numpy = train_amount.to_numpy()
#     print(train_numpy)
#     total = np.sum(train_numpy)
#     weights = [1 - np.sqrt(elements / total) for elements in train_numpy]
#     weights = np.divide(weights, np.sum(weights))
#     train_weights = torch.tensor(weights).to(torch.float32).to(device)
#     if verbose:
#         print("Done Getting Data")
#         print("=================")
#     return training_data, training_data_labels, testing_data, testing_data_labels, train_weights
