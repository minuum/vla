import h5py
import numpy as np

file_path = "/home/billy/25-1kp/vla/ROS_action/basket_dataset_v2/test/episode_20260129_011917_basket_1box_hori_left_core_medium.h5"

with h5py.File(file_path, 'r') as f:
    print(f"File: {file_path}")
    if 'language' in f:
        lang = f['language'][()]
        print(f"Language field: {lang}")
    else:
        print("No 'language' field.")
    
    if 'language_instruction' in f:
        lang_instr = f['language_instruction'][:]
        print(f"Language instruction field (first 5): {lang_instr[:5]}")
    else:
        print("No 'language_instruction' field.")

    for key in f.keys():
        print(f"Key: {key}")
