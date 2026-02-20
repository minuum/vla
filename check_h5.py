import h5py

file_path = "/home/billy/25-1kp/vla/ROS_action/basket_dataset_v2/episode_0.h5"
with h5py.File(file_path, 'r') as f:
    print("Keys:", list(f.keys()))
    if 'language_instruction' in f:
        print("Language Instruction:", f['language_instruction'][0])
    else:
        print("Language Instruction not found in H5.")
