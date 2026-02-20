import h5py
import numpy as np
import os

data_dir = "/home/billy/25-1kp/vla/ROS_action/basket_dataset_v2"
h5_files = [f for f in os.listdir(data_dir) if f.endswith('.h5')]
if not h5_files:
    print("No H5 files found.")
else:
    file_path = os.path.join(data_dir, h5_files[0])
    print(f"Checking file: {file_path}")
    with h5py.File(file_path, 'r') as f:
        print("Keys:", list(f.keys()))
        if 'language_instruction' in f:
            instruction = f['language_instruction'][()]
            if isinstance(instruction, np.ndarray):
                instruction = instruction[0]
            if isinstance(instruction, bytes):
                instruction = instruction.decode('utf-8')
            print("Instruction:", instruction)
        else:
            print("No language_instruction key found.")
