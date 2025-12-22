import h5py
from pathlib import Path

def inspect_h5_structure(h5_path):
    """Inspect the structure of an H5 file"""
    print(f"\n{'='*80}")
    print(f"파일: {h5_path.name}")
    print(f"{'='*80}")
    
    try:
        with h5py.File(h5_path, 'r') as f:
            print("최상위 키:")
            for key in f.keys():
                item = f[key]
                if isinstance(item, h5py.Dataset):
                    print(f"  - {key}: Dataset, shape={item.shape}, dtype={item.dtype}")
                    # Show first few values if small enough
                    if item.size < 100:
                        print(f"      값: {item[:]}")
                elif isinstance(item, h5py.Group):
                    print(f"  - {key}: Group")
                    # Show subkeys
                    for subkey in item.keys():
                        subitem = item[subkey]
                        if isinstance(subitem, h5py.Dataset):
                            print(f"      - {subkey}: Dataset, shape={subitem.shape}, dtype={subitem.dtype}")
    except Exception as e:
        print(f"오류: {e}")


def main():
    target_files = [
        "episode_20251203_122758_1box_hori_left_core_medium.h5",
        "episode_20251203_122846_1box_hori_left_core_medium.h5",
        "episode_20251203_122510_1box_hori_left_core_medium.h5",
        "episode_20251203_122204_1box_hori_left_core_medium.h5"
    ]
    
    dataset_dir = Path("/home/soda/vla/ROS_action/mobile_vla_dataset")
    
    print("H5 파일 구조 검사")
    
    for filename in target_files[:2]:  # Check first 2 files only
        filepath = dataset_dir / filename
        if filepath.exists():
            inspect_h5_structure(filepath)
        else:
            print(f"\n❌ 파일 없음: {filename}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
