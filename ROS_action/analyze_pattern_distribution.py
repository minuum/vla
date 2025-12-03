import h5py
from pathlib import Path
from collections import Counter

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


def extract_pattern(h5_path):
    """Extract action pattern from H5 file"""
    try:
        with h5py.File(h5_path, 'r') as f:
            if 'actions' not in f:
                return None
            
            actions = f['actions'][:]
            if len(actions) != 18:
                return None
            
            # Convert frames 1-17 to sequence
            sequence = []
            for i in range(1, 18):
                x, y, z = actions[i][0], actions[i][1], actions[i][2]
                char = get_action_char_flexible(x, y, z)
                sequence.append(char)
            
            return ' '.join(sequence)
            
    except Exception as e:
        return None


def main():
    dataset_dir = Path("/home/soda/vla/ROS_action/mobile_vla_dataset")
    
    # Get all H5 files
    h5_files = sorted(list(dataset_dir.glob("*.h5")))
    
    print("=" * 80)
    print("데이터셋 패턴 분포 분석")
    print("=" * 80)
    print(f"총 파일 수: {len(h5_files)}\n")
    
    # Extract patterns
    patterns = []
    failed = 0
    
    for h5_path in h5_files:
        pattern = extract_pattern(h5_path)
        if pattern:
            patterns.append(pattern)
        else:
            failed += 1
    
    # Count pattern distribution
    pattern_counts = Counter(patterns)
    
    print(f"분석 완료:")
    print(f"  - 성공: {len(patterns)}개")
    print(f"  - 실패: {failed}개")
    print(f"\n패턴 종류: {len(pattern_counts)}개\n")
    
    print("=" * 80)
    print("패턴 분포")
    print("=" * 80)
    
    # Sort by count (descending)
    sorted_patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)
    
    for i, (pattern, count) in enumerate(sorted_patterns, 1):
        percentage = count / len(patterns) * 100
        print(f"\n패턴 #{i}: {count}개 ({percentage:.1f}%)")
        print(f"  {pattern}")
    
    print("\n" + "=" * 80)
    print("요약")
    print("=" * 80)
    
    if len(pattern_counts) == 1:
        print("✅ 모든 파일이 동일한 패턴을 가지고 있습니다!")
        print(f"   패턴: {list(pattern_counts.keys())[0]}")
    else:
        print(f"⚠️  {len(pattern_counts)}개의 서로 다른 패턴이 발견되었습니다.")
        print(f"\n가장 많은 패턴: {sorted_patterns[0][0]}")
        print(f"  - 개수: {sorted_patterns[0][1]}개 ({sorted_patterns[0][1]/len(patterns)*100:.1f}%)")
        
        if len(sorted_patterns) > 1:
            print(f"\n두 번째로 많은 패턴: {sorted_patterns[1][0]}")
            print(f"  - 개수: {sorted_patterns[1][1]}개 ({sorted_patterns[1][1]/len(patterns)*100:.1f}%)")
    
    print("=" * 80)
    
    # Save detailed report
    report_path = dataset_dir / "pattern_distribution_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("Pattern Distribution Report\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total files analyzed: {len(patterns)}\n")
        f.write(f"Unique patterns: {len(pattern_counts)}\n\n")
        
        for i, (pattern, count) in enumerate(sorted_patterns, 1):
            percentage = count / len(patterns) * 100
            f.write(f"Pattern #{i}: {count} files ({percentage:.1f}%)\n")
            f.write(f"  {pattern}\n\n")
    
    print(f"\n💾 상세 리포트 저장: {report_path.name}")


if __name__ == "__main__":
    main()
