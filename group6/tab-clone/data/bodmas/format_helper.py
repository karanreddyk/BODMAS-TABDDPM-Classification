''' Handles the helper methods of formatter.py'''

import numpy as np
import pandas as pd
import json

# Single - Intended for large families
def single_sample(df):
    # Split into train/val/test splits, random_state allows reproducibility
    rs = 42

    # Get a minimum number for each family for training
    train = df.groupby("y_label").sample(frac=0.8, random_state=rs)
    df.drop(train.index, inplace=True)

    # Create a validation split
    val = df.groupby("y_label").sample(frac=0.5, random_state=rs)
    df.drop(val.index, inplace=True)

    # Create a testing split and merge in the gurantees
    test = df.sample(frac=1, random_state=rs)
    return train, val, test

# Double - Intended for smaller families
def double_sample(df):
    # Split into train/val/test splits, random_state allows reproducibility
    rs = 42

    # Create a training split
    train = df.groupby("y_label").sample(frac=.5, random_state=rs).index
    trainA = df.loc[train]
    trainB = df.drop(train)

    # Create a validation split
    valA = trainB.groupby("y_label").sample(frac=.5, random_state=rs)
    valB = trainA.groupby("y_label").sample(frac=.5, random_state=rs)

    # Create a testing split
    testA = trainB.drop(valA.index)
    testB = trainA.drop(valB.index)

    return trainA, valA, testA, trainB, valB, testB

def make_maps(df, size_set):
    # Remapping y_label to be subsequent
    uniques = df["y_label"].unique()

    # Create mappings in both directions
    orig_to_temp = pd.Series(data=range(len(uniques)), index=uniques)
    pd.Series(index=range(len(uniques)), data=uniques).to_json(f'data/bodmas/map_{size_set}.json')

    return orig_to_temp

def split_save(train, val, test, size_set, subgroup=""):

    storage_path = f'data/bodmas/splits/{size_set}/'
    uniques = train["y_label"].unique()

    # Save the y labels as .npy files
    np.save(f'{storage_path}y_train{subgroup}.npy', train["y_label"].to_numpy())
    np.save(f'{storage_path}y_val{subgroup}.npy', val["y_label"].to_numpy())
    np.save(f'{storage_path}y_test{subgroup}.npy', test["y_label"].to_numpy())

    # Rempve the y labels
    train.drop("y_label", axis=1, inplace=True)
    val.drop("y_label", axis=1, inplace=True)
    test.drop("y_label", axis=1, inplace=True)

    # Save the remaining features as .npy files
    np.save(f'{storage_path}X_num_train{subgroup}.npy', train.to_numpy())
    np.save(f'{storage_path}X_num_val{subgroup}.npy', val.to_numpy())
    np.save(f'{storage_path}X_num_test{subgroup}.npy', test.to_numpy())

    # Output to a json
    info = (
        f'{{"name": "bodmas", "id": "bodmas--id", "task_type": "multiclass", '
        f'"train_size": {train.shape[0]}, "val_size": {val.shape[0]}, "test_size": {test.shape[0]}, '
        f'"n_num_features": 2381, "n_cat_features": 0, "n_classes": {len(uniques)}}}'
    )

    with open(f'{storage_path}info{subgroup}.json', "w") as file:
        json.dump(json.loads(info), file, indent=4)