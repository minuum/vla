import h5py
import os
import glob

# Find first h5 file
files = glob.glob("/home/billy/25-1kp/vla/ROS_action/basket_dataset/*.h5")
if not files:
    print("No H5 files found")
    exit()

f_path = files[0]
print(f"Inspecting: {f_path}")

with h5py.File(f_path, 'r') as f:
    def print_attrs(name, obj):
        print(name)
        for key, val in obj.attrs.items():
            print(f"    Attr: {key}={val}")

    f.visititems(print_attrs)
    
    # Check specific keys if known (e.g. 'action', 'observation')
    if 'action' in f:
        print(f"Action shape: {f['action'].shape}")
    
    if 'root' in f.attrs:
        print(f"Root attr: {f.attrs['root']}")
