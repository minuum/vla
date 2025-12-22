import h5py
import numpy as np
import os
from pathlib import Path

# Configuration
DATASET_DIR = "/home/soda/vla/ROS_action/mobile_vla_dataset"
TARGET_PATTERN = ['W', 'W', 'A', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'W', 'W', 'E', 'E', 'E', 'W', 'W']

# Exact mapping from mobile_vla_data_collector.py
# x = linear_x, y = linear_y, z = angular_z
def get_action_char(x, y, z):
    """Convert action values to key character"""
    # Define threshold for comparing float values
    th = 0.1
    
    # Check if it's a stop (all near zero)
    if abs(x) < th and abs(y) < th and abs(z) < th:
        return 'S'  # Stop
    
    # Check exact matches (with tolerance)
    # W: x=1.15, y=0, z=0
    if abs(x - 1.15) < th and abs(y) < th and abs(z) < th:
        return 'W'
    
    # A: x=0, y=1.15, z=0
    if abs(x) < th and abs(y - 1.15) < th and abs(z) < th:
        return 'A'
    
    # S: x=-1.15, y=0, z=0
    if abs(x + 1.15) < th and abs(y) < th and abs(z) < th:
        return 'S'  # Backward (but named 'S')
    
    # D: x=0, y=-1.15, z=0
    if abs(x) < th and abs(y + 1.15) < th and abs(z) < th:
        return 'D'
    
    # Q: x=1.15, y=1.15, z=0
    if abs(x - 1.15) < th and abs(y - 1.15) < th and abs(z) < th:
        return 'Q'
    
    # E: x=1.15, y=-1.15, z=0
    if abs(x - 1.15) < th and abs(y + 1.15) < th and abs(z) < th:
        return 'E'
    
    # Z: x=-1.15, y=1.15, z=0
    if abs(x + 1.15) < th and abs(y - 1.15) < th and abs(z) < th:
        return 'Z'
    
    # C: x=-1.15, y=-1.15, z=0
    if abs(x + 1.15) < th and abs(y + 1.15) < th and abs(z) < th:
        return 'C'
    
    return '?'  # Unknown


def analyze_file(filepath):
    """Analyze a single H5 file and return sequence and score"""
    try:
        with h5py.File(filepath, 'r') as f:
            if 'action' not in f:
                return None, 0, "No action data"
            
            actions = f['action'][:]
            if len(actions) != 18:
                return None, 0, f"Wrong length: {len(actions)}"
            
            # Convert frames 1-17 to chars (frame 0 is stop)
            sequence = []
            for i in range(1, 18):
                x, y, z = actions[i][0], actions[i][1], actions[i][2]
                char = get_action_char(x, y, z)
                sequence.append(char)
            
            # Calculate similarity score
            score = sum(1 for i in range(len(sequence)) if i < len(TARGET_PATTERN) and sequence[i] == TARGET_PATTERN[i])
            
            return sequence, score, "OK"
            
    except Exception as e:
        return None, 0, f"Error: {str(e)}"


def main():
    files = sorted([f for f in os.listdir(DATASET_DIR) if f.endswith('.h5')])
    results = []
    
    print(f"목표 패턴: {' '.join(TARGET_PATTERN)}")
    print(f"\n총 {len(files)}개 파일 분석 중...\n")
    
    for filename in files:
        filepath = os.path.join(DATASET_DIR, filename)
        sequence, score, msg = analyze_file(filepath)
        
        if sequence:
            results.append((filename, sequence, score))
    
    # Sort by score (highest first)
    results.sort(key=lambda x: x[2], reverse=True)
    
    # Show results
    print("=" * 80)
    print("분석 결과 (유사도 높은 순서):")
    print("=" * 80)
    
    # Show perfect matches first
    perfect_matches = [r for r in results if r[2] == 17]
    if perfect_matches:
        print(f"\n✅ 완벽 매치 ({len(perfect_matches)}개):")
        for filename, seq, score in perfect_matches:
            print(f"  - {filename}")
    
    # Show top 20 closest matches
    print(f"\n📊 상위 20개 유사 파일:")
    for i, (filename, seq, score) in enumerate(results[:20]):
        seq_str = ' '.join(seq)
        match_indicator = "✅ MATCH!" if score == 17 else f"({score}/17 일치)"
        print(f"\n{i+1}. {filename} {match_indicator}")
        print(f"   목표: {' '.join(TARGET_PATTERN)}")
        print(f"   실제: {seq_str}")
        
        # Show differences
        if score < 17:
            diffs = []
            for j in range(min(len(seq), len(TARGET_PATTERN))):
                if seq[j] != TARGET_PATTERN[j]:
                    diffs.append(f"위치 {j+1}: {TARGET_PATTERN[j]} → {seq[j]}")
            if diffs:
                print(f"   차이: {', '.join(diffs)}")
    
    print("\n" + "=" * 80)
    print(f"전체 요약:")
    print(f"  - 완벽 매치: {len(perfect_matches)}개")
    print(f"  - 분석된 파일: {len(results)}개")
    if results:
        print(f"  - 최고 유사도: {results[0][2]}/17")
        print(f"  - 평균 유사도: {sum(r[2] for r in results) / len(results):.1f}/17")
    print("=" * 80)


if __name__ == "__main__":
    main()
