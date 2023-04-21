import pandas as pd
import os
import torch
import numpy as np
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
import pickle

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
        zero_df = merge[merge['class'] == 0].sample(n=max(0,zeros - int(average)))
        merge = merge.drop(zero_df.index)

        merge = merge.reset_index().drop(columns='index')
        ones_df = merge[merge['class'] == 1].sample(n=max(0,ones - int(average)))
        merge = merge.drop(ones_df.index)

        merge = merge.reset_index().drop(columns='index')
        twos_df = merge[merge['class'] == 2].sample(n=max(0,twos - int(average)))
        merge = merge.drop(twos_df.index)

        merge = merge.reset_index().drop(columns='index')
        three_df = merge[merge['class'] == 3].sample(n=max(0,threes - int(average)))
        merge = merge.drop(three_df.index)

        merge = merge.reset_index().drop(columns='index')
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
            print("Subject", i)
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
