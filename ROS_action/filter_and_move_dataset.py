import h5py
import shutil
from pathlib import Path

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


def check_h5_pattern(h5_path, target_pattern):
    """Check if H5 file matches target pattern"""
    try:
        with h5py.File(h5_path, 'r') as f:
            if 'actions' not in f:
                return False, 0, "No actions"
            
            actions = f['actions'][:]
            if len(actions) != 18:
                return False, 0, f"Wrong length: {len(actions)}"
            
            # Convert frames 1-17 to sequence
            sequence = []
            for i in range(1, 18):
                x, y, z = actions[i][0], actions[i][1], actions[i][2]
                char = get_action_char_flexible(x, y, z)
                sequence.append(char)
            
            # Check if matches
            if len(sequence) != len(target_pattern):
                return False, 0, "Length mismatch"
            
            matches = sum(1 for i in range(len(sequence)) if sequence[i] == target_pattern[i])
            is_match = (matches == len(target_pattern))
            
            return is_match, matches, sequence
            
    except Exception as e:
        return False, 0, f"Error: {str(e)}"


def main():
    dataset_dir = Path("/home/soda/vla/ROS_action/mobile_vla_dataset")
    legacy_dir = Path("/home/soda/vla/ROS_action/mobile_vla_dataset_legacy")
    target_pattern = ['W', 'W', 'A', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'W', 'W', 'E', 'E', 'E', 'W', 'W']
    
    # Create legacy dir if not exists
    legacy_dir.mkdir(exist_ok=True)
    
    # Get all H5 files
    h5_files = sorted(list(dataset_dir.glob("*.h5")))
    
    print("=" * 80)
    print("전체 데이터셋 패턴 매칭 및 정리")
    print("=" * 80)
    print(f"목표 패턴: {' '.join(target_pattern)}")
    print(f"총 H5 파일 수: {len(h5_files)}")
    print(f"\n스캔 중...\n")
    
    matching_files = []
    non_matching_files = []
    
    # Scan all files
    for i, h5_path in enumerate(h5_files):
        if (i + 1) % 50 == 0:
            print(f"진행 중: {i+1}/{len(h5_files)}...")
        
        is_match, score, info = check_h5_pattern(h5_path, target_pattern)
        
        if is_match:
            matching_files.append(h5_path.name)
        else:
            non_matching_files.append((h5_path.name, score, info))
    
    print(f"\n완료: {len(h5_files)}개 파일 스캔")
    print("=" * 80)
    print("분석 결과")
    print("=" * 80)
    print(f"✅ 패턴 일치: {len(matching_files)}개")
    print(f"❌ 패턴 불일치: {len(non_matching_files)}개")
    
    # Show matching files
    if matching_files:
        print(f"\n일치하는 파일 (유지):")
        for filename in matching_files[:20]:  # Show first 20
            print(f"  - {filename}")
        if len(matching_files) > 20:
            print(f"  ... 외 {len(matching_files) - 20}개 더")
    
    # Move non-matching files
    if non_matching_files:
        print(f"\n불일치 파일을 legacy로 이동 중...")
        moved_count = 0
        
        for filename, score, info in non_matching_files:
            h5_path = dataset_dir / filename
            
            # Move H5 file
            try:
                shutil.move(str(h5_path), str(legacy_dir / filename))
                moved_count += 1
                
                # Also move related files (JSON, CSV, directories)
                base_name = h5_path.stem
                
                # Move JSON files
                for json_file in dataset_dir.glob(f"{base_name}*.json"):
                    shutil.move(str(json_file), str(legacy_dir / json_file.name))
                
                # Move CSV files
                for csv_file in dataset_dir.glob(f"{base_name}*.csv"):
                    shutil.move(str(csv_file), str(legacy_dir / csv_file.name))
                
                # Move directories
                dir_path = dataset_dir / base_name
                if dir_path.exists() and dir_path.is_dir():
                    shutil.move(str(dir_path), str(legacy_dir / base_name))
                    
            except Exception as e:
                print(f"  ⚠️  이동 실패: {filename} - {e}")
        
        print(f"✅ 이동 완료: {moved_count}개 파일")
    
    print("\n" + "=" * 80)
    print("최종 결과")
    print("=" * 80)
    print(f"✅ mobile_vla_dataset에 남은 파일: {len(matching_files)}개")
    print(f"📦 mobile_vla_dataset_legacy로 이동: {len(non_matching_files)}개")
    print(f"\n목표 수량: 1000개")
    print(f"남은 수량: {1000 - len(matching_files)}개")
    print(f"진행률: {len(matching_files)/1000*100:.1f}%")
    print("=" * 80)
    
    # Save summary
    summary_path = dataset_dir / "dataset_cleanup_summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"Dataset Cleanup Summary\n")
        f.write(f"{'='*80}\n")
        f.write(f"Target Pattern: {' '.join(target_pattern)}\n\n")
        f.write(f"Matching files kept: {len(matching_files)}\n")
        f.write(f"Non-matching files moved to legacy: {len(non_matching_files)}\n\n")
        f.write(f"Remaining target: {1000 - len(matching_files)}\n\n")
        
        f.write(f"Kept files:\n")
        for filename in matching_files:
            f.write(f"  - {filename}\n")
    
    print(f"\n💾 요약 저장: {summary_path.name}")


if __name__ == "__main__":
    main()
