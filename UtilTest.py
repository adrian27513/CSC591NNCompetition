from util import get_balanced_data
import pickle
import pandas as pd

train, train_label, test, test_label = get_balanced_data(1, True)

for t in train:
    print(t.shape)