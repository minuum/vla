import h5py
import numpy as np

file_path = '/home/billy/25-1kp/vla/ROS_action/basket_dataset_v2/episode_20260129_011330_basket_1box_hori_left_core_medium.h5'
with h5py.File(file_path, 'r') as f:
    print(f"Keys in H5: {list(f.keys())}")
    if 'language_instruction' in f:
        lang = f['language_instruction'][0]
        if isinstance(lang, bytes):
            lang = lang.decode('utf-8')
        print(f"Language Instruction: {lang}")
    else:
        print("No language_instruction found in H5.")
