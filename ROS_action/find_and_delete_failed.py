import h5py
from pathlib import Path
import os

def get_action_char_flexible(x, y, z):
    """Convert action values to key character with flexible threshold"""
    zero_th = 0.3
    pos_low = 0.5
    pos_high = 1.5
    neg_low = -1.5
    neg_high = -0.5
    
    if abs(x) < zero_th and abs(y) < zero_th and abs(z) < zero_th:
        return 'STOP'
    if pos_low <= x <= pos_high and abs(y) < zero_th and abs(z) < zero_th:
        return 'W'
    if abs(x) < zero_th and pos_low <= y <= pos_high and abs(z) < zero_th:
        return 'A'
    if neg_low <= x <= neg_high and abs(y) < zero_th and abs(z) < zero_th:
        return 'S'
    if abs(x) < zero_th and neg_low <= y <= neg_high and abs(z) < zero_th:
        return 'D'
    if pos_low <= x <= pos_high and pos_low <= y <= pos_high and abs(z) < zero_th:
        return 'Q'
    if pos_low <= x <= pos_high and neg_low <= y <= neg_high and abs(z) < zero_th:
        return 'E'
    if neg_low <= x <= neg_high and pos_low <= y <= pos_high and abs(z) < zero_th:
        return 'Z'
    if neg_low <= x <= neg_high and neg_low <= y <= neg_high and abs(z) < zero_th:
        return 'C'
    
    return f'?({x:.2f},{y:.2f},{z:.2f})'


def check_file(h5_path):
    """Check if file can be analyzed"""
    try:
        with h5py.File(h5_path, 'r') as f:
            if 'actions' not in f:
                return False, "No actions dataset"
            
            actions = f['actions'][:]
            if len(actions) != 18:
                return False, f"Wrong length: {len(actions)}"
            
            # Try to extract pattern
            sequence = []
            for i in range(1, 18):
                x, y, z = actions[i][0], actions[i][1], actions[i][2]
                char = get_action_char_flexible(x, y, z)
                sequence.append(char)
            
            return True, ' '.join(sequence)
            
    except Exception as e:
        return False, f"Error: {str(e)}"


def main():
    dataset_dir = Path("/home/soda/vla/ROS_action/mobile_vla_dataset")
    
    # Get all H5 files
    h5_files = sorted(list(dataset_dir.glob("*.h5")))
    
    print("=" * 80)
    print("분석 실패 파일 찾기")
    print("=" * 80)
    print(f"총 파일 수: {len(h5_files)}\n")
    
    failed_files = []
    
    for h5_path in h5_files:
        success, info = check_file(h5_path)
        if not success:
            failed_files.append((h5_path, info))
            print(f"❌ {h5_path.name}")
            print(f"   이유: {info}\n")
    
    print("=" * 80)
    print(f"분석 실패 파일: {len(failed_files)}개")
    print("=" * 80)
    
    if failed_files:
        print("\n삭제할 파일:")
        for h5_path, reason in failed_files:
            print(f"  - {h5_path.name} ({reason})")
        
        # Delete files
        print("\n파일 삭제 중...")
        for h5_path, reason in failed_files:
            try:
                # Delete H5 file
                os.remove(h5_path)
                print(f"✅ 삭제: {h5_path.name}")
                
                # Also delete related files
                base_name = h5_path.stem
                
                # Delete JSON files
                for json_file in dataset_dir.glob(f"{base_name}*.json"):
                    os.remove(json_file)
                    print(f"   └─ {json_file.name}")
                
                # Delete CSV files
                for csv_file in dataset_dir.glob(f"{base_name}*.csv"):
                    os.remove(csv_file)
                    print(f"   └─ {csv_file.name}")
                
                # Delete directories
                dir_path = dataset_dir / base_name
                if dir_path.exists() and dir_path.is_dir():
                    import shutil
                    shutil.rmtree(dir_path)
                    print(f"   └─ {base_name}/ (디렉토리)")
                    
            except Exception as e:
                print(f"⚠️  삭제 실패: {h5_path.name} - {e}")
        
        print(f"\n✅ 총 {len(failed_files)}개 파일 삭제 완료")
    else:
        print("\n✅ 분석 실패한 파일이 없습니다.")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
