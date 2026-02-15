#!/usr/bin/env python3
"""
수집된 데이터의 trajectory(가이드) 종류를 분석하는 스크립트
"""
import h5py
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

def infer_key_from_action(action: Dict[str, float]) -> str:
    """액션에서 키 추론"""
    lx, ly, az = action['linear_x'], action['linear_y'], action['angular_z']
    
    # 정확한 매칭 (1.15 또는 -1.15)
    if abs(lx - 1.15) < 0.1 and abs(ly) < 0.1 and abs(az) < 0.1:
        return 'W'
    elif abs(lx) < 0.1 and abs(ly - 1.15) < 0.1 and abs(az) < 0.1:
        return 'A'
    elif abs(lx + 1.15) < 0.1 and abs(ly) < 0.1 and abs(az) < 0.1:
        return 'S'
    elif abs(lx) < 0.1 and abs(ly + 1.15) < 0.1 and abs(az) < 0.1:
        return 'D'
    elif abs(lx - 1.15) < 0.1 and abs(ly - 1.15) < 0.1 and abs(az) < 0.1:
        return 'Q'
    elif abs(lx - 1.15) < 0.1 and abs(ly + 1.15) < 0.1 and abs(az) < 0.1:
        return 'E'
    elif abs(lx + 1.15) < 0.1 and abs(ly - 1.15) < 0.1 and abs(az) < 0.1:
        return 'Z'
    elif abs(lx + 1.15) < 0.1 and abs(ly + 1.15) < 0.1 and abs(az) < 0.1:
        return 'C'
    elif abs(lx) < 0.1 and abs(ly) < 0.1 and abs(az - 1.15) < 0.1:
        return 'R'
    elif abs(lx) < 0.1 and abs(ly) < 0.1 and abs(az + 1.15) < 0.1:
        return 'T'
    elif abs(lx) < 0.1 and abs(ly) < 0.1 and abs(az) < 0.1:
        return 'SPACE'
    else:
        return 'UNKNOWN'

def extract_trajectory_from_h5(file_path: Path) -> Tuple[str, List[str]]:
    """H5 파일에서 trajectory 추출"""
    try:
        with h5py.File(file_path, 'r') as f:
            actions = f['actions'][:]
            action_event_types = f['action_event_types'][:]
            
            # 문자열 디코딩
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
            
            # trajectory를 문자열로 변환 (비교용)
            trajectory_str = " ".join(trajectory)
            
            return trajectory_str, trajectory
    except Exception as e:
        print(f"❌ {file_path.name} 분석 실패: {e}")
        return None, None

def analyze_dataset(dataset_dir: Path):
    """데이터셋 전체 분석"""
    h5_files = list(dataset_dir.glob("*.h5"))
    
    print(f"📊 총 {len(h5_files)}개의 H5 파일 발견")
    print("=" * 80)
    
    # trajectory별 통계
    trajectory_stats = defaultdict(list)  # trajectory_str -> [episode_names]
    scenario_trajectory_stats = defaultdict(lambda: defaultdict(list))  # scenario -> trajectory_str -> [episode_names]
    
    # 시나리오별 통계
    scenario_stats = defaultdict(int)
    
    for h5_file in sorted(h5_files):
        trajectory_str, trajectory = extract_trajectory_from_h5(h5_file)
        if trajectory_str is None:
            continue
        
        episode_name = h5_file.stem
        trajectory_stats[trajectory_str].append(episode_name)
        
        # 시나리오 추출 (파일명에서 추출)
        # 형식: episode_YYYYMMDD_HHMMSS_1box_hori_right_core_medium
        scenario = None
        parts = episode_name.split('_')
        for i, part in enumerate(parts):
            if part in ['1box', '2box']:
                # 다음 부분이 hori/vert일 수 있으므로 그 다음을 확인
                if i + 2 < len(parts):
                    direction = parts[i + 2]
                    if direction in ['left', 'right']:
                        scenario = f"{part}_{direction}"
                        break
                elif i + 1 < len(parts):
                    direction = parts[i + 1]
                    if direction in ['left', 'right']:
                        scenario = f"{part}_{direction}"
                        break
        
        if scenario:
            scenario_stats[scenario] += 1
            scenario_trajectory_stats[scenario][trajectory_str].append(episode_name)
    
    # 결과 출력
    print("\n📋 시나리오별 수집 통계:")
    print("=" * 80)
    for scenario in sorted(scenario_stats.keys()):
        count = scenario_stats[scenario]
        print(f"  {scenario}: {count}개")
    
    print("\n📊 Trajectory 종류별 통계:")
    print("=" * 80)
    sorted_trajectories = sorted(trajectory_stats.items(), key=lambda x: len(x[1]), reverse=True)
    
    for idx, (trajectory_str, episodes) in enumerate(sorted_trajectories, 1):
        print(f"\n[{idx}] {len(episodes)}개 에피소드")
        print(f"    Trajectory: {trajectory_str}")
        print(f"    길이: {len(trajectory_str.split())} 액션")
        if len(episodes) <= 10:
            print(f"    에피소드: {', '.join([e.split('_')[2] for e in episodes[:10]])}")
        else:
            print(f"    에피소드 (처음 10개): {', '.join([e.split('_')[2] for e in episodes[:10]])} ...")
    
    print("\n📋 시나리오별 Trajectory 분포:")
    print("=" * 80)
    for scenario in sorted(scenario_trajectory_stats.keys()):
        print(f"\n🎯 {scenario}:")
        traj_dict = scenario_trajectory_stats[scenario]
        sorted_trajs = sorted(traj_dict.items(), key=lambda x: len(x[1]), reverse=True)
        
        for trajectory_str, episodes in sorted_trajs:
            print(f"  • {len(episodes)}개: {trajectory_str}")
            if len(episodes) <= 5:
                print(f"    → {', '.join([e.split('_')[2] for e in episodes])}")
    
    # 요약
    print("\n" + "=" * 80)
    print("📊 요약:")
    print(f"  총 에피소드: {len(h5_files)}개")
    print(f"  고유 Trajectory 종류: {len(trajectory_stats)}개")
    print(f"  시나리오 종류: {len(scenario_stats)}개")
    print("=" * 80)
    
    return trajectory_stats, scenario_trajectory_stats

if __name__ == "__main__":
    dataset_dir = Path("/home/soda/vla/ROS_action/mobile_vla_dataset")
    analyze_dataset(dataset_dir)

