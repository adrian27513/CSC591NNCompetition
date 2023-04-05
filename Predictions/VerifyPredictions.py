import glob
import os
import numpy as np

# Specifying the folder containg the data
folderPred = os.getcwd()

# Getting the list of subject files and sorting them
L = glob.glob(folderPred + '/subject_*__y.csv')
L.sort()

# Checking that the number of files and the names are correct
assert len(L)==4
assert os.path.basename(L[0])=='subject_009_01__y.csv'
assert os.path.basename(L[1])=='subject_010_01__y.csv'
assert os.path.basename(L[2])=='subject_011_01__y.csv'
assert os.path.basename(L[3])=='subject_012_01__y.csv'

# Specifying the data expected lengths of the predictions
predLen = [9498, 12270, 12940, 11330]

# Loading the data
for i in range(0, 4):
    # Loading the predictions
    pred = np.genfromtxt(L[i], delimiter=',')

    assert len(pred.shape) == 1  # Checking that this is a single column
    assert pred.shape[0] == predLen[i]  # Making sure you have the correct number of data points
    assert min(pred) >= 0 and max(pred) <= 3  # Prediction should be either 0, 1, 2, or 3
