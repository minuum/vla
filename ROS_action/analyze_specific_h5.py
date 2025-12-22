import h5py
import json
import numpy as np
from pathlib import Path

def h5_to_json(h5_path):
    """Convert H5 file to JSON format"""
    try:
        with h5py.File(h5_path, 'r') as f:
            # Extract data
            frames_data = []
            
            if 'action' in f and 'images' in f:
                actions = f['action'][:]
                images = f['images'][:]
                
                # Get event types if available
                event_types = []
                if 'event_type' in f:
                    event_types = [et.decode('utf-8') if isinstance(et, bytes) else str(et) 
                                  for et in f['event_type'][:]]
                
                # Build frames
                for i in range(len(actions)):
                    frame_data = {
                        "frame_index": i,
                        "action": {
                            "x": float(actions[i][0]),
                            "y": float(actions[i][1]),
                            "z": float(actions[i][2])
                        },
                        "event_type": event_types[i] if i < len(event_types) else "unknown",
                        "image_file": f"frame_{i:04d}.png"
                    }
                    frames_data.append(frame_data)
                
                # Create JSON structure
                json_data = {
                    "episode_name": h5_path.stem,
                    "total_frames": len(actions),
                    "frames": frames_data
                }
                
                return json_data
            else:
                return None
                
    except Exception as e:
        print(f"Error processing {h5_path}: {e}")
        return None


def get_action_char_flexible(x, y, z):
    """
    Convert action values to key character with flexible threshold (0.5~1.5)
    """
    # Near zero threshold
    zero_th = 0.3
    # Positive action threshold (0.5~1.5)
    pos_low = 0.5
    pos_high = 1.5
    # Negative action threshold
    neg_low = -1.5
    neg_high = -0.5
    
    # Check if it's a stop (all near zero)
    if abs(x) < zero_th and abs(y) < zero_th and abs(z) < zero_th:
        return 'STOP'
    
    # W: x in [0.5, 1.5], y near 0, z near 0
    if pos_low <= x <= pos_high and abs(y) < zero_th and abs(z) < zero_th:
        return 'W'
    
    # A: x near 0, y in [0.5, 1.5], z near 0
    if abs(x) < zero_th and pos_low <= y <= pos_high and abs(z) < zero_th:
        return 'A'
    
    # S: x in [-1.5, -0.5], y near 0, z near 0
    if neg_low <= x <= neg_high and abs(y) < zero_th and abs(z) < zero_th:
        return 'S'
    
    # D: x near 0, y in [-1.5, -0.5], z near 0
    if abs(x) < zero_th and neg_low <= y <= neg_high and abs(z) < zero_th:
        return 'D'
    
    # Q: x in [0.5, 1.5], y in [0.5, 1.5], z near 0
    if pos_low <= x <= pos_high and pos_low <= y <= pos_high and abs(z) < zero_th:
        return 'Q'
    
    # E: x in [0.5, 1.5], y in [-1.5, -0.5], z near 0
    if pos_low <= x <= pos_high and neg_low <= y <= neg_high and abs(z) < zero_th:
        return 'E'
    
    # Z: x in [-1.5, -0.5], y in [0.5, 1.5], z near 0
    if neg_low <= x <= neg_high and pos_low <= y <= pos_high and abs(z) < zero_th:
        return 'Z'
    
    # C: x in [-1.5, -0.5], y in [-1.5, -0.5], z near 0
    if neg_low <= x <= neg_high and neg_low <= y <= neg_high and abs(z) < zero_th:
        return 'C'
    
    return f'?({x:.3f},{y:.3f},{z:.3f})'


def analyze_file(h5_path):
    """Convert H5 to JSON and analyze pattern"""
    json_data = h5_to_json(h5_path)
    
    if not json_data:
        return None, None, "Failed to convert"
    
    frames = json_data['frames']
    if len(frames) < 18:
        return json_data, None, f"Only {len(frames)} frames"
    
    # Extract action sequence (frames 1-17, frame 0 is stop)
    sequence = []
    for i in range(1, min(18, len(frames))):
        action = frames[i]['action']
        char = get_action_char_flexible(action['x'], action['y'], action['z'])
        sequence.append(char)
    
    return json_data, sequence, "OK"


def main():
    # Target files
    target_files = [
        "episode_20251203_122758_1box_hori_left_core_medium.h5",
        "episode_20251203_122846_1box_hori_left_core_medium.h5",
        "episode_20251203_122510_1box_hori_left_core_medium.h5",
        "episode_20251203_122204_1box_hori_left_core_medium.h5"
    ]
    
    dataset_dir = Path("/home/soda/vla/ROS_action/mobile_vla_dataset")
    target_pattern = ['W', 'W', 'A', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'W', 'W', 'E', 'E', 'E', 'W', 'W']
    
    print("=" * 80)
    print("H5 파일 분석 (JSON 변환 포함)")
    print("=" * 80)
    print(f"목표 패턴: {' '.join(target_pattern)}\n")
    
    for filename in target_files:
        filepath = dataset_dir / filename
        
        if not filepath.exists():
            print(f"\n❌ 파일 없음: {filename}")
            continue
        
        print(f"\n{'='*80}")
        print(f"📄 파일: {filename}")
        print(f"{'='*80}")
        
        json_data, sequence, status = analyze_file(filepath)
        
        if json_data:
            # Save JSON
            json_path = filepath.with_suffix('.json').with_name(filepath.stem + '_converted.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            print(f"✅ JSON 저장: {json_path.name}")
        
        if sequence:
            print(f"\n액션 시퀀스 (프레임 1-17):")
            print(f"  실제: {' '.join(sequence)}")
            print(f"  목표: {' '.join(target_pattern)}")
            
            # Calculate match
            matches = sum(1 for i in range(min(len(sequence), len(target_pattern))) 
                         if sequence[i] == target_pattern[i])
            print(f"\n  매칭: {matches}/17 ({matches/17*100:.1f}%)")
            
            if matches == 17:
                print(f"  ✅ 완벽 매치!")
            else:
                # Show differences
                diffs = []
                for i in range(min(len(sequence), len(target_pattern))):
                    if sequence[i] != target_pattern[i]:
                        diffs.append(f"위치{i+1}[{target_pattern[i]}→{sequence[i]}]")
                if diffs:
                    print(f"  차이점: {', '.join(diffs[:10])}")
            
            # Show detailed action values for first few frames
            if json_data:
                print(f"\n  상세 액션 값 (처음 5개):")
                for i in range(1, min(6, len(json_data['frames']))):
                    frame = json_data['frames'][i]
                    action = frame['action']
                    char = sequence[i-1] if i-1 < len(sequence) else '?'
                    print(f"    프레임 {i}: x={action['x']:.3f}, y={action['y']:.3f}, z={action['z']:.3f} → {char}")
        else:
            print(f"❌ 분석 실패: {status}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
