import os
import requests
import numpy as np
import pandas as pd
import torch
import datetime

from scipy.io import savemat, loadmat
from scipy.stats import zscore
from models import ResNetMapper
from memory_profiler import profile

N_DIV_OPP = 100
N_DIV_MHEALTH = 100
N_DIV_URFALL = 10
N_LABEL_DIV_OPP = 15
N_LABEL_DIV_MHEALTH = 9
N_LABEL_DIV_URFALL = 9


def fill_nan(matrix):
    """Fill NaN values with the value of the same column from previous row

    Args:
        matrix: a 2-d numpy matrix
    Return:
        A 2-d numpy matrix with NaN values filled
    """
    m = matrix
    np.nan_to_num(x=m[0, :], copy=False, nan=0.0)
    for row in range(1, m.shape[0]):
        for col in range(m.shape[1]):
            if np.isnan(m[row, col]):
                m[row, col] = m[row-1, col]
    return m



def resample_class(class_data, max_size):
    if len(class_data) < max_size:
        return class_data[np.random.choice(class_data.shape[0], max_size, replace=True)]
    else:
        return class_data

def rebalance_opp_data(data):
    labels = data[:, 115]

    labeled_mask = labels != 0
    labeled_data = data[labeled_mask]

    classes = np.unique(labeled_data[:, 115])
    class_dict = {label: labeled_data[labeled_data[:, 115] == label] for label in classes}

    # Find the size of the largest class
    max_size = max(len(class_data) for class_data in class_dict.values())

    balanced_class_list = [resample_class(class_dict[label], max_size) for label in classes]

    balanced_labeled_data = np.vstack(balanced_class_list)

    # Print the class distribution in the balanced labeled data
    unique, counts = np.unique(balanced_labeled_data[:, 115], return_counts=True)
    
    print("labeled data counts")
    print(dict(zip(unique, counts)))
    
    # Combine with unlabeled data if needed
    unlabeled_data = data[~labeled_mask]
    final_data = np.vstack([balanced_labeled_data, unlabeled_data]) # include unlabeled data
    #final_data = np.vstack([balanced_labeled_data]) # do not include unlabeled data
    np.random.shuffle(final_data)
    return final_data
    #return balanced_labeled_data

def gen_opp_balanced(data_path):
    """Generates training, validating, and testing data from Opp datasets

    Args:
        data_path: the path of the Opportunity challenge dataset

    Returns:
        None
    """
    acce_columns = [i-1 for i in range(2, 41)]
    acce_columns.extend([46, 47, 48, 55, 56, 57, 64, 65, 66, 73, 74,
                         75, 85, 86, 87, 88, 89, 90, 101, 102, 103, 104, 105, 106, ])
    gyro_columns = [40, 41, 42, 49, 50, 51,
                    58, 59, 60, 67, 68, 69, 66, 67, 68, ]
    acce_gyro_columns = acce_columns.extend(gyro_columns) #include both acce and gyro columns
    # Loads the run 2 from subject 1 as validating data
    data_valid = np.loadtxt(os.path.join(data_path, "opp", "S1-ADL2.dat"))
    x_valid_acce = fill_nan(data_valid[:, acce_columns])
    x_valid_gyro = fill_nan(data_valid[:, gyro_columns])
    y_valid = data_valid[:, 115]
    

    # Loads the runs 4 and 5 from subjects 2 and 3 as testing data
    runs_test = []
    idxs_test = []
    for r in [4, 5]:
        for s in [2, 3]:
            runs_test.append(np.loadtxt(os.path.join(
                data_path, "opp", f"S{s}-ADL{r}.dat")))
            idxs_test.append((r, s))
    data_test = np.concatenate(runs_test)
    x_test_acce = fill_nan(data_test[:, acce_columns])
    x_test_gyro = fill_nan(data_test[:, gyro_columns])
    y_test = data_test[:, 115]

    # if 1, fill nan on the whole training data before rebalance
    # else, rebalance first then fill_nan on x_train_acce and x_train_gyro
    fill_nan_before_rebalance = 1; 
    # Loads the remaining runs as training data
    runs_train = []
    for r in range(1, 6):
        for s in range(1, 5):
            if (r, s) not in idxs_test:
                runs_train.append(np.loadtxt(os.path.join(
                    data_path, "opp", f"S{s}-ADL{r}.dat")))
    data_train = np.concatenate(runs_train)
    if fill_nan_before_rebalance == 1: 
        #fill nan on all the acce gyro columns before rebalancing training data
        data_train = fill_nan(data_train)
        data_train = rebalance_opp_data(data_train)
        x_train_acce = data_train[:, acce_columns]
        x_train_gyro = data_train[:, gyro_columns]
    else: 
        data_train = rebalance_opp_data(data_train)
        x_train_acce = fill_nan(data_train[:, acce_columns])
        x_train_gyro = fill_nan(data_train[:, gyro_columns])

    y_train = data_train[:, 115]

    # Changes labels to (0, 1, ...)
    unique_y = list(set(y_train).union(set(y_valid)).union(set(y_test)))
    unique_y.sort()

    print(unique_y)

    print(f"Number of values in y_train: {len(y_train)}")  # Should print 10000
    print(f"Distinct values in y_train: {np.unique(y_train, return_counts=True)}") 
    unique_values, class_counts = np.unique(y_train, return_counts=True)
    print(class_counts)

    y_map = {}
    for idx, y in enumerate(unique_y):
        y_map[y] = idx
    y_train = np.vectorize(y_map.get)(y_train)
    y_valid = np.vectorize(y_map.get)(y_valid)
    y_test = np.vectorize(y_map.get)(y_test)

    mdic = {}
    mdic["x_train_acce"] = x_train_acce
    mdic["x_train_gyro"] = x_train_gyro
    mdic["y_train"] = np.squeeze(y_train)
    mdic["x_valid_acce"] = x_valid_acce
    mdic["x_valid_gyro"] = x_valid_gyro
    mdic["y_valid"] = np.squeeze(y_valid)  # This only has 17 classes
    mdic["x_test_acce"] = x_test_acce
    mdic["x_test_gyro"] = x_test_gyro
    mdic["y_test"] = np.squeeze(y_test)

    savemat(os.path.join(data_path, "opp", "opp.mat"), mdic)



if __name__ == "__main__":
    # gen_opp("data")
    gen_opp_balanced("data")
    # gen_mhealth("data")
    # download_UR_fall()
    # gen_ur_fall("data")
    # gen_ur_fall("download")
    pass

