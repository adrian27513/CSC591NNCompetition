import pandas as pd
import os
import torch
import numpy as np
import torch.nn.functional as F
import random
device = torch.device("cuda:0")

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

def get_data(window, verbose):
    training_list = list(set(e.split('__')[0] for e in os.listdir('TrainingData')))
    if verbose:
        print(training_list)
        print(len(training_list))
    training_data = []
    training_data_labels = []
    testing_data = []
    testing_data_labels = []
    split = 0.2

    train_amount = pd.Series([0, 0, 0, 0], index=[0.0, 1.0, 2.0, 3.0])

    for i, data in enumerate(training_list):
        if (verbose):
            print("Subject", i, ":", data)
        aligned = align_data(data)
        df_data = aligned.drop(columns=['label'])
        df_label = aligned['label']

        train_amount = train_amount.add(df_label.iloc[:int(len(aligned.index) * split)].value_counts(), fill_value=0)
        test_split = int(len(aligned.index) * split)
        test_start = random.randint(0, len(aligned.index) - test_split)
        # Train
        training = df_data.drop(df_data.index[test_start:test_start+test_split]).to_numpy()
        training_labels = df_label.drop(df_data.index[test_start:test_start+test_split]).to_numpy()
        sub_windows = (-(window - 1) + np.expand_dims(np.arange(window), 0) + np.expand_dims(np.arange(len(training)), 0).T)[window - 1:]
        sub_features = torch.from_numpy(training[sub_windows])
        padded = []
        for idx in range(1, window):
            feature = torch.from_numpy(training[max(0,idx-window):idx])
            length = feature.shape[0]
            feature_padded = F.pad(feature, (0,0,window - length,0), value=-1)
            padded.append(feature_padded)
        if window != 1:
            stacked_padded = torch.stack(padded)
            train_features = torch.cat((stacked_padded, sub_features), 0).to(torch.float32)
        else:
            train_features = sub_features.to(torch.float32)
        train_labels = torch.tensor(training_labels).to(torch.long)
        training_data.append(train_features)
        training_data_labels.append(train_labels)

        # Test
        testing = df_data.iloc[test_start:test_start+test_split, :].to_numpy()
        testing_labels = df_label.to_numpy()[test_start:test_start+test_split]
        sub_windows = (-(window - 1) + np.expand_dims(np.arange(window), 0) + np.expand_dims(np.arange(len(testing)), 0).T)[window - 1:]
        sub_features = torch.from_numpy(testing[sub_windows])

        padded = []
        for idx in range(1, window):
            feature = torch.from_numpy(testing[max(0, idx - window):idx])
            length = feature.shape[0]
            feature_padded = F.pad(feature, (0, 0, window - length, 0), value=0)
            padded.append(feature_padded)
        if window != 1:
            stacked_padded = torch.stack(padded)
            test_features = torch.cat((stacked_padded, sub_features), 0).to(torch.float32)
        else:
            test_features = sub_features.to(torch.float32)
        test_labels = torch.tensor(testing_labels).to(torch.long)
        testing_data.append(test_features)
        testing_data_labels.append(test_labels)

    train_numpy = train_amount.to_numpy()
    total = np.sum(train_numpy)
    weights = [1-np.sqrt(elements/total) for elements in train_numpy]
    weights = np.divide(weights, np.sum(weights))
    train_weights = torch.tensor(weights).to(torch.float32).to(device)
    if verbose:
        print("Done Getting Data")
        print("=================")
    return training_data, training_data_labels, testing_data, testing_data_labels, train_weights
