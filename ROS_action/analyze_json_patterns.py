import json
import os
from pathlib import Path

# Configuration
DATASET_DIR = "/home/soda/vla/ROS_action/mobile_vla_dataset"
TARGET_PATTERN = ['W', 'W', 'A', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'W', 'W', 'E', 'E', 'E', 'W', 'W']

def get_action_char(x, y, z):
    """Convert action values to key character based on mobile_vla_data_collector.py mappings"""
    th = 0.1
    
    # Stop
    if abs(x) < th and abs(y) < th and abs(z) < th:
        return 'STOP'
    
    # W: x=1.15, y=0, z=0
    if abs(x - 1.15) < th and abs(y) < th and abs(z) < th:
        return 'W'
    
    # A: x=0, y=1.15, z=0
    if abs(x) < th and abs(y - 1.15) < th and abs(z) < th:
        return 'A'
    
    # S: x=-1.15, y=0, z=0 (backward)
    if abs(x + 1.15) < th and abs(y) < th and abs(z) < th:
        return 'S'
    
    # D: x=0, y=-1.15, z=0
    if abs(x) < th and abs(y + 1.15) < th and abs(z) < th:
        return 'D'
    
    # Q: x=1.15, y=1.15, z=0
    if abs(x - 1.15) < th and abs(y - 1.15) < th and abs(z) < th:
        return 'Q'
    
    # E: x=1.15, y=-1.15, z=0
    if abs(x - 1.15) < th and abs(y + 1.15) < th and abs(z) < th:
        return 'E'
    
    return f'?({x:.2f},{y:.2f},{z:.2f})'


def analyze_json_file(filepath):
    """Analyze JSON file and return sequence and score"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'frames' not in data:
            return None, 0, "No frames data"
        
        frames = data['frames']
        if len(frames) != 18:
            return None, 0, f"Wrong frame count: {len(frames)}"
        
        # Convert frames 1-17 to chars (frame 0 is episode_start/stop)
        sequence = []
        for i in range(1, 18):
            action = frames[i]['action']
            x, y, z = action['x'], action['y'], action['z']
            char = get_action_char(x, y, z)
            sequence.append(char)
        
        # Calculate similarity score (only count exact matches)
        score = sum(1 for i in range(len(sequence)) if i < len(TARGET_PATTERN) and sequence[i] == TARGET_PATTERN[i])
        
        return sequence, score, "OK"
        
    except Exception as e:
        return None, 0, f"Error: {str(e)}"


def main():
    # Find all JSON files with _data.json suffix
    files = sorted([f for f in os.listdir(DATASET_DIR) if f.endswith('_data.json')])
    results = []
    
    print(f"목표 패턴: {' '.join(TARGET_PATTERN)}")
    print(f"\n총 {len(files)}개 JSON 파일 분석 중...\n")
    
    for filename in files:
        filepath = os.path.join(DATASET_DIR, filename)
        sequence, score, msg = analyze_json_file(filepath)
        
        if sequence:
            results.append((filename, sequence, score))
        elif msg != "OK":
            print(f"⚠️  {filename}: {msg}")
    
    # Sort by score (highest first)
    results.sort(key=lambda x: x[2], reverse=True)
    
    # Show results
    print("\n" + "=" * 80)
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
                    diffs.append(f"위치{j+1}: 예상[{TARGET_PATTERN[j]}] 실제[{seq[j]}]")
            if diffs:
                print(f"   차이: {', '.join(diffs[:5])}" + ("..." if len(diffs) > 5 else ""))
    
    print("\n" + "=" * 80)
    print(f"전체 요약:")
    print(f"  - 완벽 매치: {len(perfect_matches)}개")
    print(f"  - 분석된 파일: {len(results)}개")
    if results:
        print(f"  - 최고 유사도: {results[0][2]}/17")
        print(f"  - 평균 유사도: {sum(r[2] for r in results) / len(results):.1f}/17")
    
    # Show distribution of scores
    if results:
        score_dist = {}
        for _, _, score in results:
            score_dist[score] = score_dist.get(score, 0) + 1
        print(f"\n  - 유사도 분포:")
        for score in sorted(score_dist.keys(), reverse=True):
            count = score_dist[score]
            print(f"      {score}/17 일치: {count}개")
    
    print("=" * 80)
    
    # If there are any matches >= 15, save them to a file
    high_matches = [r for r in results if r[2] >= 15]
    if high_matches:
        print(f"\n💾 유사도 15/17 이상인 파일 목록을 'high_similarity_files.txt'에 저장합니다...")
        with open('/home/soda/vla/ROS_action/high_similarity_files.txt', 'w') as f:
            f.write("# Files with similarity >= 15/17\n")
            for filename, seq, score in high_matches:
                f.write(f"{filename} ({score}/17)\n")
        print(f"✅ {len(high_matches)}개 파일 목록 저장 완료")


if __name__ == "__main__":
    main()
