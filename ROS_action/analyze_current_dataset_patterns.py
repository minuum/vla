#!/usr/bin/env python3
"""
현재 mobile_vla_dataset의 가이드 패턴 분포 분석
"""

import h5py
import re
from pathlib import Path
from collections import Counter, defaultdict

dataset_dir = Path("/home/soda/vla/ROS_action/mobile_vla_dataset")

# H5 파일 목록 가져오기
h5_files = sorted(dataset_dir.glob("episode_*.h5"))

if not h5_files:
    print("❌ H5 파일을 찾을 수 없습니다.")
    exit(1)

print(f"📊 현재 데이터셋 분석: {len(h5_files)}개 파일\n")

# 패턴 추출
pattern_counter = Counter()
scenario_patterns = defaultdict(Counter)

for h5_file in h5_files:
    try:
        with h5py.File(h5_file, 'r') as f:
            # actions 데이터에서 패턴 추출
            if 'actions' in f:
                actions = f['actions'][:]
                
                # 액션을 키로 변환
                key_sequence = []
                for action in actions:
                    lx, ly, az = action[0], action[1], action[2]
                    
                    # 액션을 키로 매핑
                    if abs(lx - 1.15) < 0.1 and abs(ly) < 0.1 and abs(az) < 0.1:
                        key = 'W'
                    elif abs(lx) < 0.1 and abs(ly - 1.15) < 0.1 and abs(az) < 0.1:
                        key = 'A'
                    elif abs(lx + 1.15) < 0.1 and abs(ly) < 0.1 and abs(az) < 0.1:
                        key = 'S'
                    elif abs(lx) < 0.1 and abs(ly + 1.15) < 0.1 and abs(az) < 0.1:
                        key = 'D'
                    elif abs(lx - 1.15) < 0.1 and abs(ly - 1.15) < 0.1 and abs(az) < 0.1:
                        key = 'Q'
                    elif abs(lx - 1.15) < 0.1 and abs(ly + 1.15) < 0.1 and abs(az) < 0.1:
                        key = 'E'
                    elif abs(lx + 1.15) < 0.1 and abs(ly - 1.15) < 0.1 and abs(az) < 0.1:
                        key = 'Z'
                    elif abs(lx + 1.15) < 0.1 and abs(ly + 1.15) < 0.1 and abs(az) < 0.1:
                        key = 'C'
                    elif abs(lx) < 0.1 and abs(ly) < 0.1 and abs(az - 1.0) < 0.1:
                        key = 'R'
                    elif abs(lx) < 0.1 and abs(ly) < 0.1 and abs(az + 1.0) < 0.1:
                        key = 'T'
                    else:
                        key = '?'  # 알 수 없는 액션
                    
                    key_sequence.append(key)
                
                # 패턴 문자열 생성
                pattern = ' '.join(key_sequence)
                pattern_counter[pattern] += 1
                
                # 파일명에서 시나리오 추출
                match = re.search(r'episode_\d+_(\w+)_', h5_file.name)
                if match:
                    scenario = match.group(1)
                    scenario_patterns[scenario][pattern] += 1
                
    except Exception as e:
        print(f"⚠️  {h5_file.name} 처리 중 오류: {e}")

print("=" * 80)
print("📋 전체 패턴 분포 (상위 10개)")
print("=" * 80)
for pattern, count in pattern_counter.most_common(10):
    percentage = (count / len(h5_files)) * 100
    print(f"  {pattern}")
    print(f"    → {count}회 ({percentage:.1f}%)")
    print()

print("=" * 80)
print("📋 시나리오별 패턴 분포")
print("=" * 80)
for scenario, patterns in sorted(scenario_patterns.items()):
    print(f"\n🎯 {scenario}:")
    total = sum(patterns.values())
    for pattern, count in patterns.most_common(5):
        percentage = (count / total) * 100
        print(f"  {pattern}")
        print(f"    → {count}회 ({percentage:.1f}%)")

print("\n" + "=" * 80)
print("✅ 분석 완료")
print("=" * 80)

