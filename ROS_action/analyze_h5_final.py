import h5py
import json
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


def analyze_h5_file(h5_path):
    """Analyze H5 file and return sequence"""
    try:
        with h5py.File(h5_path, 'r') as f:
            if 'actions' not in f:
                return None, "No actions dataset"
            
            actions = f['actions'][:]
            event_types = [et.decode('utf-8') if isinstance(et, bytes) else str(et) 
                          for et in f['action_event_types'][:]] if 'action_event_types' in f else []
            
            # Convert frames 1-17 to sequence (frame 0 is stop)
            sequence = []
            frames_detail = []
            
            for i in range(1, len(actions)):
                x, y, z = actions[i][0], actions[i][1], actions[i][2]
                char = get_action_char_flexible(x, y, z)
                sequence.append(char)
                frames_detail.append({
                    'frame': i,
                    'x': float(x),
                    'y': float(y),
                    'z': float(z),
                    'char': char,
                    'event': event_types[i] if i < len(event_types) else 'unknown'
                })
            
            return sequence, frames_detail, "OK"
            
    except Exception as e:
        return None, None, f"Error: {e}"


def main():
    target_files = [
        "episode_20251203_122758_1box_hori_left_core_medium.h5",
        "episode_20251203_122846_1box_hori_left_core_medium.h5",
        "episode_20251203_122510_1box_hori_left_core_medium.h5",
        "episode_20251203_122204_1box_hori_left_core_medium.h5"
    ]
    
    dataset_dir = Path("/home/soda/vla/ROS_action/mobile_vla_dataset")
    target_pattern = ['W', 'W', 'A', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'W', 'W', 'E', 'E', 'E', 'W', 'W']
    
    print("=" * 80)
    print("H5 파일 상세 분석")
    print("=" * 80)
    print(f"목표 패턴 (17개): {' '.join(target_pattern)}\n")
    
    all_match = True
    matching_files = []
    
    for filename in target_files:
        filepath = dataset_dir / filename
        
        if not filepath.exists():
            print(f"\n❌ 파일 없음: {filename}")
            continue
        
        print(f"\n{'='*80}")
        print(f"📄 {filename}")
        print(f"{'='*80}")
        
        sequence, frames_detail, status = analyze_h5_file(filepath)
        
        if sequence:
            # Save JSON
            json_data = {
                "filename": filename,
                "target_pattern": target_pattern,
                "actual_sequence": sequence,
                "frames": frames_detail
            }
            
            json_path = dataset_dir / f"{filepath.stem}_analysis.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            print(f"실제 시퀀스 ({len(sequence)}개): {' '.join(sequence)}")
            print(f"목표 패턴  (17개): {' '.join(target_pattern)}")
            
            # Calculate match
            if len(sequence) == 17:
                matches = sum(1 for i in range(17) if sequence[i] == target_pattern[i])
                print(f"\n매칭률: {matches}/17 ({matches/17*100:.1f}%)")
                
                if matches == 17:
                    print(f"✅ 완벽 매치!")
                    matching_files.append(filename)
                else:
                    all_match = False
                    # Show differences
                    diffs = []
                    for i in range(17):
                        if sequence[i] != target_pattern[i]:
                            diffs.append(f"위치{i+1}: 목표[{target_pattern[i]}] 실제[{sequence[i]}]")
                    print(f"\n차이점 ({len(diffs)}개):")
                    for diff in diffs:
                        print(f"  - {diff}")
            else:
                all_match = False
                print(f"\n⚠️  길이 불일치: 실제 {len(sequence)}개 vs 목표 17개")
            
            # Show first 5 frames detail
            print(f"\n상세 액션 (처음 5개):")
            for frame in frames_detail[:5]:
                print(f"  프레임 {frame['frame']}: "
                      f"x={frame['x']:.2f}, y={frame['y']:.2f}, z={frame['z']:.2f} "
                      f"→ {frame['char']}")
            
            print(f"\n💾 JSON 저장: {json_path.name}")
        else:
            print(f"❌ 분석 실패: {status}")
            all_match = False
    
    print("\n" + "=" * 80)
    print("전체 요약")
    print("=" * 80)
    if matching_files:
        print(f"✅ 목표 패턴과 일치하는 파일: {len(matching_files)}개")
        for f in matching_files:
            print(f"  - {f}")
    else:
        print(f"❌ 목표 패턴과 일치하는 파일 없음")
    
    if all_match and len(matching_files) == len(target_files):
        print(f"\n🎉 모든 파일이 목표 패턴과 일치합니다!")
    print("=" * 80)


if __name__ == "__main__":
    main()
