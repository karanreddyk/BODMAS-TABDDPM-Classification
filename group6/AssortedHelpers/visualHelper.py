''' Prints the shape of an npy file'''

import numpy as np
import pandas as pd
import sys

folder = sys.argv[1]

Xn_train = np.load(folder + "X_num_train.npy")
X_num_train = pd.DataFrame(Xn_train)
Xn_val = np.load(folder + "X_num_val.npy")
X_num_val = pd.DataFrame(Xn_val)
Xn_test = np.load(folder + "X_num_test.npy")
X_num_test = pd.DataFrame(Xn_test)
try:
    Xc_train = np.load(folder + "X_cat_train.npy")
    X_cat_train = pd.DataFrame(Xc_train)
    Xc_val = np.load(folder + "X_cat_val.npy")
    X_cat_val = pd.DataFrame(Xc_val)
    Xc_test = np.load(folder + "X_cat_test.npy")
    X_cat_test = pd.DataFrame(Xc_test)
    ynp_train = np.load(folder + "y_train.npy")
    y_train = pd.DataFrame(ynp_train)
    ynp_val = np.load(folder + "y_val.npy")
    y_val = pd.DataFrame(ynp_val)
    ynp_test = np.load(folder + "y_test.npy")
    y_test = pd.DataFrame(ynp_test)

    print(f'Shape Printing\n\nX_num:\n Train: {X_num_train.shape}\n Val: {X_num_val.shape}\n Test: {X_num_test.shape}\n\nX_cat:\n Train: {X_cat_train.shape}\n Val: {X_cat_val.shape}\n Test: {X_cat_test.shape}\n\ny:\n Train: {y_train.shape}\n Val: {y_val.shape}\n Test: {y_test.shape}\n')
except:
    ynp_train = np.load(folder + "y_train.npy")
    y_train = pd.DataFrame(ynp_train)
    ynp_val = np.load(folder + "y_val.npy")
    y_val = pd.DataFrame(ynp_val)
    ynp_test = np.load(folder + "y_test.npy")
    y_test = pd.DataFrame(ynp_test)

    print(f'Shape Printing\n\nX_num:\n Train: {X_num_train.shape}\n Val: {X_num_val.shape}\n Test: {X_num_test.shape}\n\ny:\n Train: {y_train.shape}\n Val: {y_val.shape}\n Test: {y_test.shape}\n')
