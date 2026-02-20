import h5py

file_path = "/home/billy/25-1kp/vla/ROS_action/basket_dataset_v2/episode_20260129_011330_basket_1box_hori_left_core_medium.h5"
with h5py.File(file_path, 'r') as f:
    print(f"Keys: {list(f.keys())}")
    if 'language_instruction' in f:
        print(f"language_instruction: {f['language_instruction'][0]}")
    else:
        print("language_instruction key not found.")
