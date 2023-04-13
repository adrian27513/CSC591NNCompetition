from util import get_data

training_data, training_data_labels, testing_data, testing_data_labels, train_weights = get_data(1, True)

print(training_data[0].shape)
print(training_data_labels[0].shape)
print(train_weights)
