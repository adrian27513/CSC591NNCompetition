import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
import random
import time
from model import RNN
from util import get_data
from ax.service.managed_loop import optimize
from sklearn.metrics import confusion_matrix


device = torch.device("cuda:0")


def train_model(model, parameters, training_data, training_data_labels, criterion):
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=parameters.get("learning_rate", 0.01), betas=(0.9, 0.99))

    num_epochs = parameters.get("num_epochs", 10)
    subject_list = list(zip(training_data, training_data_labels))

    for epoch in range(num_epochs):
        start = time.time()
        total_loss = []
        for batch in subject_list:
            mini_batch_size = parameters.get("mini_batch_size", 512)
            input_data = batch[0]
            # print("Input Shape", input_data.shape)
            input_label = batch[1]
            batch_input = torch.split(input_data, mini_batch_size)
            batch_label = torch.split(input_label, mini_batch_size)
            for batch_in, batch_label in zip(batch_input, batch_label):
                test_in = batch_in.to(device=device)
                test_in.to(device)
                test_label = batch_label.to(device=device)
                output = model(test_in)
                model.zero_grad()
                loss = criterion(output, test_label)
                total_loss.append(loss.item())
                loss.backward()
                optimizer.step()
        print("Average Loss:", np.average(total_loss))
        iter_time = time.time() - start
        print("Iter Time:", round(iter_time, 3), "seconds")
        hours_left = iter_time * (num_epochs - epoch) / 3600
        hours = int(hours_left)
        minutes = int((hours_left * 60) % 60)
        seconds = int((hours_left * 3600) % 60)
        print("Epoch:", epoch)
        print("Estimated Time: ", hours, "hours", minutes, "minutes", seconds, "seconds")
    return model.to(device='cpu')


def init_model(parameters):
    return RNN(6, 4, parameters.get("dropout", 0.2), parameters.get("lstm_hidden", 200),
               parameters.get("linear_hidden", 50),parameters.get("lstm_layers", 1))


def eval_model(model, parameters, testing_data, testing_data_labels, criterion):
    model.to(device)
    model.eval()
    with torch.no_grad():
        test_losses = []
        pred_sum = 0
        total = 0
        actual = []
        predicted = []
        mini_batch_size = parameters.get("mini_batch_size", 512)
        for subject_data in list(zip(testing_data, testing_data_labels)):
            input_data = subject_data[0].to(device=device)
            test_label = subject_data[1].to(device=device)
            batch_input = torch.split(input_data, mini_batch_size)
            batch_label = torch.split(test_label, mini_batch_size)
            for batch_in, batch_label in zip(batch_input, batch_label):
                test_in = batch_in.to(device=device)
                test_label = batch_label.to(device=device)
                output = model(test_in)
                loss = criterion(output, test_label)
                test_losses.append(loss.item())
                max_pred = torch.argmax(output, dim=1)
                actual.extend(test_label.tolist())
                predicted.extend(max_pred.tolist())
                equal = torch.eq(test_label, max_pred)
                pred_sum += torch.sum(equal).item()
                total += equal.shape[0]

        confusion = confusion_matrix(actual, predicted)
        print(confusion)
        print("Loss:", np.average(test_losses))
        return pred_sum / total


def train_evaluate(parameters):
    print("Parameters:",parameters)
    training_data, training_data_labels, testing_data, testing_data_labels, weight = get_data(
        parameters.get("window_size", 5), verbose=False)
    untrained_model = init_model(parameters)
    criterion = nn.CrossEntropyLoss()
    trained_model = train_model(model=untrained_model, parameters=parameters, training_data=training_data,
                                training_data_labels=training_data_labels, criterion=criterion)
    accuracy = eval_model(model=trained_model, parameters=parameters, testing_data=testing_data,
                          testing_data_labels=testing_data_labels, criterion=criterion)
    print("Accuracy:", accuracy)
    return accuracy

best_parameters, values, experiment, model = optimize(
    parameters=[
        {"name": "dropout", "type": "range", "bounds": [0.0, 0.9]},
        {"name": "window_size", "type": "range", "bounds": [1, 20]},
        {"name": "linear_hidden", "type": "range", "bounds": [1, 500]},
        {"name": "lstm_hidden", "type": "range", "bounds": [1, 500]},
        # {"name": "lstm_layers", "type": "range", "bounds": [1, 3]}
    ],
    evaluation_function=train_evaluate,
    objective_name="Accuracy",
    total_trials=20
)

print(best_parameters)
means, covariances = values
print(means)
print(covariances)