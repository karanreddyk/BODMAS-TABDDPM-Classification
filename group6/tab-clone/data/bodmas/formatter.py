''' Converts a normalized .npy file into the npy files that match
    TabDDPM's epected format'''

import numpy as np
import pandas as pd
import format_helper as fh

# Load from the complete feature vector list
DATA_DIR = f'../BODMAS/code/multiple_data/g6data/'
X = np.load(f'{DATA_DIR}X_train_real_seed0.npy')
y = np.load(f'{DATA_DIR}y_train_real_seed0.npy')
df = pd.DataFrame(X)
df['y_label'] = y

print("Load: Success")

# Settings 
breaks = [40, 100] # Small < [0] <= Medium < [1] <= Large
sample_methods = {"s": False, "m": False, "l": False} # Determines which sampling method to use

# Get list of labels to keep
family_counts = df['y_label'].value_counts()

s_valid = family_counts[family_counts < breaks[0]].index
sdf = df[df['y_label'].isin(s_valid)]

m_valid = family_counts[(family_counts >= breaks[0]) & (family_counts < breaks[1])].index
mdf = df[df['y_label'].isin(m_valid)]

l_valid = family_counts[family_counts >= breaks[1]].index
ldf = df[df['y_label'].isin(l_valid)]

print(f'Filter: Success')

# Create the maps
s_map = fh.make_maps(sdf, "s")
m_map = fh.make_maps(mdf, "m")
l_map = fh.make_maps(ldf, "l")

# This stops an annoying warning
sdf = sdf.copy()
mdf = mdf.copy()
ldf = ldf.copy()

# Apply the maps to the dataframes
sdf["y_label"] = sdf["y_label"].map(s_map)
mdf["y_label"] = mdf["y_label"].map(m_map)
ldf["y_label"] = ldf["y_label"].map(l_map)

print(f'Mapping: Success')

# Maps size to the correct dataframe
df_map = {"s": sdf, "m": mdf, "l": ldf}

# Sample and Save for each group
for key, method in sample_methods.items():
    if method:
        trainA, valA, testA, trainB, valB, testB = fh.double_sample(df_map[key])
        fh.split_save(trainA, valA, testA, key, "A")
        fh.split_save(trainB, valB, testB, key, "B")
    else:
        train, val, test = fh.single_sample(df_map[key])
        fh.split_save(train, val, test, key)

print("Save:Success")