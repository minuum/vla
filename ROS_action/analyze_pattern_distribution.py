#!/usr/bin/env python3
"""
시나리오별, 거리별, 패턴별 궤적 패턴 분포 분석
"""
import h5py
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np

def infer_key_from_action(action):
    lx, ly, az = action['linear_x'], action['linear_y'], action['angular_z']
    if abs(az) > 0.1:
        return 'R' if az > 0 else 'T'
    if abs(lx) < 0.1 and abs(ly) < 0.1:
        return 'SPACE'
    if lx > 0.1 and abs(ly) <= 0.1:
        return 'W'
    if lx < -0.1 and abs(ly) <= 0.1:
        return 'S'
    if ly > 0.1 and abs(lx) <= 0.1:
        return 'A'
    if ly < -0.1 and abs(lx) <= 0.1:
        return 'D'
    if lx > 0.1 and ly > 0.1:
        return 'Q'
    if lx > 0.1 and ly < -0.1:
        return 'E'
    if lx < -0.1 and ly > 0.1:
        return 'Z'
    if lx < -0.1 and ly < -0.1:
        return 'C'
    return 'UNK'

def extract_trajectory(h5_file):
    try:
        with h5py.File(h5_file, 'r') as f:
            actions = f['actions'][:]
            action_event_types = f['action_event_types'][:]
            if isinstance(action_event_types[0], bytes):
                action_event_types = [e.decode('utf-8') for e in action_event_types]
            trajectory = []
            for idx, ev in enumerate(action_event_types):
                if ev == 'start_action':
                    action = {
                        'linear_x': float(actions[idx][0]),
                        'linear_y': float(actions[idx][1]),
                        'angular_z': float(actions[idx][2])
                    }
                    key = infer_key_from_action(action)
                    trajectory.append(key)
            return ' '.join(trajectory)
    except Exception as e:
        print(f"❌ {h5_file.name} 분석 실패: {e}")
        return None

# 모든 h5 파일 분석
dataset_dir = Path('/home/soda/vla/ROS_action/mobile_vla_dataset')
h5_files = list(dataset_dir.glob('*.h5'))

# 시나리오별, 거리별, 패턴별로 그룹화
scenario_pattern_stats = defaultdict(lambda: defaultdict(int))

for h5_file in h5_files:
    name = h5_file.stem
    # episode_20251119_080007_1box_hori_right_core_medium 형식
    parts = name.split('_')
    
    # 시나리오 추출 (1box_left, 1box_right 등)
    scenario = None
    distance = None
    pattern = None
    
    for i, part in enumerate(parts):
        if part in ['1box', '2box']:
            if i + 2 < len(parts):
                direction = parts[i + 2]
                if direction in ['left', 'right']:
                    scenario = f"{part}_{direction}"
                    break
    
    # 거리 추출 (close, medium, far)
    for part in parts:
        if part in ['close', 'medium', 'far']:
            distance = part
            break
    
    # 패턴 추출 (core, variant)
    for part in parts:
        if part in ['core', 'variant']:
            pattern = part
            break
    
    if scenario and distance and pattern:
        trajectory = extract_trajectory(h5_file)
        if trajectory:
            key = f"{scenario}__{pattern}__{distance}"
            scenario_pattern_stats[key][trajectory] += 1

print("=" * 80)
print("📊 시나리오 × 패턴 × 거리별 궤적 패턴 분석")
print("=" * 80)

for key in sorted(scenario_pattern_stats.keys()):
    print(f"\n🎯 {key}:")
    traj_dict = scenario_pattern_stats[key]
    sorted_trajs = sorted(traj_dict.items(), key=lambda x: x[1], reverse=True)
    
    total = sum(traj_dict.values())
    for trajectory, count in sorted_trajs:
        percentage = (count / total) * 100
        print(f"  • {count}개 ({percentage:.1f}%): {trajectory}")
        if count == total:
            print(f"    ✅ 모든 에피소드가 동일한 패턴!")

print("\n" + "=" * 80)
print("📋 요약: 각 조합별 가장 많은 패턴")
print("=" * 80)

# 각 조합별로 가장 많은 패턴 확인
for key in sorted(scenario_pattern_stats.keys()):
    traj_dict = scenario_pattern_stats[key]
    if not traj_dict:
        continue
    
    sorted_trajs = sorted(traj_dict.items(), key=lambda x: x[1], reverse=True)
    most_common = sorted_trajs[0]
    total = sum(traj_dict.values())
    
    print(f"\n{key}:")
    print(f"  총 {total}개")
    print(f"  가장 많은 패턴: {most_common[1]}개")
    print(f"    궤적: {most_common[0]}")
    if len(sorted_trajs) > 1:
        print(f"  다른 패턴: {len(sorted_trajs) - 1}개")
        for traj, count in sorted_trajs[1:]:
            print(f"    - {count}개: {traj}")

