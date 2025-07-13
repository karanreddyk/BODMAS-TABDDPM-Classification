# Takes the small, medium, and large synthetic datasets and merges them
import numpy as np
import pandas as pd
import sys
import json

# Output variables
OUTPUT_PATH = f'../BODMAS/code/multiple_data/g6data/'

# Function that returns a pd.DataFrame with remapped y_label
def build_df(size_str, file_str):
    # Load relevant data
    load_path = f'exp/bodmas/ddpm_{file_str}f_tune_best/'
    data_X = np.load(f'{load_path}X_num_train.npy')
    data_y = np.load(f'{load_path}y_train.npy')
    with open(f'data/bodmas/map_{size_str}.json') as f:
        rev_map = json.load(f)

    # Create and return new dataframe
    built_df = pd.DataFrame(data_X)
    built_df['y_label'] = [rev_map[str(int(y))] for y in data_y]

    return built_df

# Load the synthetic data into dataframes
sdf = build_df("s", "s")
mdf = build_df("m", "m")
ldf = build_df("l", "l")

# Combine the dataframes
df = pd.concat([sdf, mdf, ldf], ignore_index=True)

# Separate the labels
df_y = df["y_label"].to_numpy()
df.drop("y_label", axis=1, inplace=True)

# Save for use with BODMAS
np.save(f'{OUTPUT_PATH}synth_X_num_train', df.to_numpy())
np.save(f'{OUTPUT_PATH}synth_y_train', df_y)