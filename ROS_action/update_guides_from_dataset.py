#!/usr/bin/env python3
"""
현재 데이터셋에서 가장 많은 패턴을 추출하여 가이드로 업데이트
"""
import h5py
from pathlib import Path
from collections import defaultdict, Counter
import json
import numpy as np

def infer_key_from_action(action):
    """액션에서 키 추론"""
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
    """H5 파일에서 궤적 추출"""
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

def normalize_to_18_keys(keys, target_length=17):
    """17개 액션으로 정규화 (초기 프레임 1개 + 17개 액션 = 18 프레임)"""
    normalized = list(keys[:target_length])
    if len(normalized) < target_length:
        normalized += ['SPACE'] * (target_length - len(normalized))
    return normalized

def main():
    dataset_dir = Path('/home/soda/vla/ROS_action/mobile_vla_dataset')
    core_pattern_file = dataset_dir / "core_patterns.json"
    
    h5_files = list(dataset_dir.glob('*.h5'))
    print(f"📊 총 {len(h5_files)}개 파일 분석 중...\n")
    
    # 조합별 패턴 통계
    combo_pattern_stats = defaultdict(lambda: defaultdict(int))
    
    for h5_file in h5_files:
        name = h5_file.stem
        parts = name.split('_')
        
        # 시나리오, 거리, 패턴 추출
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
        
        for part in parts:
            if part in ['close', 'medium', 'far']:
                distance = part
                break
        
        for part in parts:
            if part in ['core', 'variant']:
                pattern = part
                break
        
        if scenario and distance and pattern:
            trajectory = extract_trajectory(h5_file)
            if trajectory:
                combo_key = f"{scenario}__{pattern}__{distance}"
                combo_pattern_stats[combo_key][trajectory] += 1
    
    # 각 조합별로 가장 많은 패턴 찾기
    print("=" * 80)
    print("📊 조합별 최다 패턴 분석")
    print("=" * 80)
    
    updated_patterns = {}
    
    for combo_key in sorted(combo_pattern_stats.keys()):
        pattern_dict = combo_pattern_stats[combo_key]
        if not pattern_dict:
            continue
        
        sorted_patterns = sorted(pattern_dict.items(), key=lambda x: x[1], reverse=True)
        most_common = sorted_patterns[0]
        total = sum(pattern_dict.values())
        
        print(f"\n{combo_key}:")
        print(f"  총 {total}개 파일")
        print(f"  최다 패턴: {most_common[1]}개 ({most_common[1]/total*100:.1f}%)")
        print(f"  궤적: {most_common[0]}")
        
        # 키 리스트로 변환 (SPACE는 그대로 유지)
        keys = most_common[0].split()
        # 17개로 정규화
        normalized = normalize_to_18_keys(keys, target_length=17)
        # 끝에 SPACE만 남았을 경우 제거
        while normalized and normalized[-1] == 'SPACE':
            normalized.pop()
        
        # core 패턴만 저장
        if 'core' in combo_key:
            updated_patterns[combo_key] = normalized
            print(f"  ✅ 가이드로 저장: {len(normalized)}개 액션")
    
    # 기존 가이드 로드
    existing_patterns = {}
    if core_pattern_file.exists():
        try:
            with open(core_pattern_file, 'r', encoding='utf-8') as f:
                existing_patterns = json.load(f)
            print(f"\n📋 기존 가이드: {len(existing_patterns)}개")
        except Exception as e:
            print(f"⚠️ 기존 가이드 로드 실패: {e}")
    
    # 업데이트된 패턴과 기존 패턴 병합
    final_patterns = existing_patterns.copy()
    for key, pattern in updated_patterns.items():
        old_pattern = final_patterns.get(key, [])
        if old_pattern != pattern:
            print(f"\n🔄 업데이트: {key}")
            if old_pattern:
                old_str = " ".join([k.upper() for k in old_pattern])
                print(f"  기존: {old_str}")
            new_str = " ".join([k.upper() for k in pattern])
            print(f"  신규: {new_str}")
        final_patterns[key] = pattern
    
    # 저장
    print("\n" + "=" * 80)
    print("💾 가이드 저장 중...")
    print("=" * 80)
    
    try:
        core_pattern_file.parent.mkdir(parents=True, exist_ok=True)
        with open(core_pattern_file, 'w', encoding='utf-8') as f:
            json.dump(final_patterns, f, indent=2, ensure_ascii=False)
        print(f"✅ 가이드 저장 완료: {core_pattern_file}")
        print(f"📊 총 {len(final_patterns)}개 가이드 저장됨")
        
        print("\n📋 저장된 가이드 목록:")
        for key in sorted(final_patterns.keys()):
            pattern = final_patterns[key]
            pattern_str = " ".join([k.upper() for k in pattern])
            print(f"  {key}: {pattern_str} ({len(pattern)}개 액션)")
    except Exception as e:
        print(f"❌ 가이드 저장 실패: {e}")

if __name__ == "__main__":
    main()

