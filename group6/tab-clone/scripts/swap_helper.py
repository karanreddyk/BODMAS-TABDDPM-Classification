# Changes the active split in the bodmas folder, saves from lots of renaming
# Run from the tab-clone folder
# To activate the small A set use: python scripts/swap_helper.py s A

import sys
import shutil

FILES = ["X_num_train", "X_num_val", "X_num_test", "y_train", "y_val", "y_test"]

# Activates a previously made split
def swap():

    # Use size to make the path
    size = sys.argv[1]
    if size not in ("s", "m", "l"):
        print("Error: Unrecognized Size")
        return False
    READ_PATH = f'data/bodmas/splits/{size}/'
    WRITE_PATH = f'data/bodmas/'

    # Use this to figure out more specific activation
    try:
        target = sys.argv[2]
    except IndexError:
        target = ""

    if target not in ("", "A", "B"):
        print("Error: Unrecognized Target")
        return False
    
    # Move each npy file
    for file_name in FILES:
        shutil.copy(f'{READ_PATH}{file_name}{target}.npy', f'{WRITE_PATH}{file_name}.npy')

    # Move the json file
    shutil.copy(f'{READ_PATH}info{target}.json', f'{WRITE_PATH}info.json')

    # Move the config file
    shutil.copy(f'exp/bodmas/base_configs/config_{size}{target}.toml', f'exp/bodmas/config.toml')



try:
    swap()
except FileNotFoundError:
    print("Error: Missing File")