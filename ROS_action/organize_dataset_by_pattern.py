#!/usr/bin/env python3
"""
주요 패턴 파일만 남기고 나머지는 legacy 디렉토리로 백업
"""
import h5py
import shutil
from pathlib import Path
from collections import defaultdict

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

def main():
    dataset_dir = Path('/home/soda/vla/ROS_action/mobile_vla_dataset')
    legacy_dir = Path('/home/soda/vla/ROS_action/mobile_vla_dataset_legacy')
    
    # Legacy 디렉토리 생성
    legacy_dir.mkdir(exist_ok=True)
    print(f"📁 Legacy 디렉토리 생성: {legacy_dir}")
    
    # 주요 패턴 정의
    main_patterns = {
        '1box_left__core__medium': 'W W W A Q Q Q Q Q Q Q Q W W W W Q',
        '1box_right__core__medium': 'W W W D E E E E E E W W W W Q Q Q',
    }
    
    h5_files = list(dataset_dir.glob('*.h5'))
    print(f"\n📊 총 {len(h5_files)}개 파일 분석 시작...\n")
    
    keep_files = []
    move_files = []
    
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
        
        if not (scenario and distance and pattern):
            # 조합을 추출할 수 없으면 legacy로 이동
            move_files.append((h5_file, 'unknown_combination'))
            continue
        
        combo_key = f"{scenario}__{pattern}__{distance}"
        
        # 주요 패턴인지 확인
        if combo_key in main_patterns:
            trajectory = extract_trajectory(h5_file)
            if trajectory == main_patterns[combo_key]:
                keep_files.append((h5_file, combo_key, trajectory))
            else:
                move_files.append((h5_file, f"{combo_key}_different_pattern"))
        else:
            # 주요 패턴 조합이 아니면 legacy로 이동
            move_files.append((h5_file, combo_key))
    
    # 결과 출력
    print("=" * 80)
    print("📋 정리 결과")
    print("=" * 80)
    print(f"\n✅ 유지할 파일: {len(keep_files)}개")
    for h5_file, combo_key, traj in keep_files:
        print(f"  • {h5_file.name} ({combo_key})")
    
    print(f"\n📦 Legacy로 이동할 파일: {len(move_files)}개")
    move_by_reason = defaultdict(list)
    for h5_file, reason in move_files:
        move_by_reason[reason].append(h5_file)
    
    for reason, files in sorted(move_by_reason.items()):
        print(f"\n  {reason}: {len(files)}개")
        for f in files[:5]:  # 처음 5개만 표시
            print(f"    - {f.name}")
        if len(files) > 5:
            print(f"    ... 외 {len(files) - 5}개")
    
    # 자동 실행 (사용자 확인 생략)
    print("\n" + "=" * 80)
    print("🔄 자동으로 파일 이동을 시작합니다...")
    
    # 파일 이동
    print("\n🔄 파일 이동 중...")
    moved_count = 0
    for h5_file, reason in move_files:
        try:
            dest = legacy_dir / h5_file.name
            shutil.move(str(h5_file), str(dest))
            moved_count += 1
            if moved_count % 10 == 0:
                print(f"  이동 중... {moved_count}/{len(move_files)}")
        except Exception as e:
            print(f"❌ {h5_file.name} 이동 실패: {e}")
    
    print(f"\n✅ 완료! {moved_count}개 파일을 legacy 디렉토리로 이동했습니다.")
    print(f"📁 Legacy 디렉토리: {legacy_dir}")
    print(f"✅ 유지된 파일: {len(keep_files)}개")

if __name__ == "__main__":
    main()

