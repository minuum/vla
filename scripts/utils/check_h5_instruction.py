import h5py
import glob

# 실제 존재하는 H5 파일 찾기
h5_files = glob.glob("/home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset/*.h5")
if not h5_files:
    print("No H5 files found!")
    exit(1)

h5_file = h5_files[0]  # 첫 번째 파일 사용

with h5py.File(h5_file, 'r') as f:
    print("=" * 60)
    print(f"File: {h5_file.split('/')[-1]}")
    print("=" * 60)
    
    # Attributes
    print("\n[Attributes]")
    for key in f.attrs.keys():
        val = f.attrs[key]
        if isinstance(val, bytes):
            val = val.decode('utf-8')
        print(f"  {key}: {val}")
    
    # Datasets
    print("\n[Datasets]")
    for key in f.keys():
        print(f"  {key}: shape={f[key].shape}, dtype={f[key].dtype}")
    
    # Check if there's instruction in attrs
    if 'instruction' in f.attrs:
        instr = f.attrs['instruction']
        if isinstance(instr, bytes):
            instr = instr.decode('utf-8')
        print(f"\n[INSTRUCTION FOUND IN ATTRS]")
        print(f"  Value: '{instr}'")
    else:
        print(f"\n[NO INSTRUCTION IN ATTRS]")
    
    # Check if there's instruction dataset
    if 'instruction' in f.keys():
        print(f"\n[INSTRUCTION FOUND IN DATASET]")
        print(f"  Value: {f['instruction'][()]}")
    else:
        print(f"\n[NO INSTRUCTION IN DATASET]")
        
    print("\n" + "=" * 60)
