"""
🔍 액션 값 상세 분석
모든 프레임의 액션 값을 확인하여 데이터 문제 진단
"""

import h5py
import numpy as np
from pathlib import Path
import os

def analyze_action_values():
    """액션 값 상세 분석"""
    
    print("🔍 액션 값 상세 분석")
    print("=" * 50)
    
    data_path = '../../ROS_action/mobile_vla_dataset'
    
    if os.path.isdir(data_path):
        h5_files = list(Path(data_path).glob("*.h5"))
    else:
        h5_files = [data_path]
    
    print(f"📊 분석할 H5 파일 수: {len(h5_files)}")
    
    # 몇 개 파일만 상세 분석
    sample_files = h5_files[:5]
    
    for h5_file in sample_files:
        print(f"\n🔍 {h5_file.name} 분석:")
        print("-" * 40)
        
        try:
            with h5py.File(h5_file, 'r') as f:
                if 'actions' in f:
                    actions = f['actions'][:]  # [18, 3]
                    
                    print(f"📊 액션 전체 shape: {actions.shape}")
                    print(f"📊 액션 데이터 타입: {actions.dtype}")
                    
                    # 모든 프레임의 액션 값 출력
                    print(f"📋 모든 프레임 액션 값:")
                    for frame_idx in range(actions.shape[0]):
                        action = actions[frame_idx]
                        print(f"   프레임 {frame_idx:2d}: [{action[0]:8.6f}, {action[1]:8.6f}, {action[2]:8.6f}]")
                    
                    # 통계 정보
                    print(f"📊 액션 통계:")
                    print(f"   - 최솟값: [{actions[:, 0].min():8.6f}, {actions[:, 1].min():8.6f}, {actions[:, 2].min():8.6f}]")
                    print(f"   - 최댓값: [{actions[:, 0].max():8.6f}, {actions[:, 1].max():8.6f}, {actions[:, 2].max():8.6f}]")
                    print(f"   - 평균값: [{actions[:, 0].mean():8.6f}, {actions[:, 1].mean():8.6f}, {actions[:, 2].mean():8.6f}]")
                    print(f"   - 표준편차: [{actions[:, 0].std():8.6f}, {actions[:, 1].std():8.6f}, {actions[:, 2].std():8.6f}]")
                    
                    # 0이 아닌 액션 개수
                    non_zero_actions = np.count_nonzero(actions, axis=0)
                    print(f"   - 0이 아닌 액션 개수: X축={non_zero_actions[0]}, Y축={non_zero_actions[1]}, Z축={non_zero_actions[2]}")
                    
                    # 첫 프레임과 마지막 프레임 비교
                    first_action = actions[0]
                    last_action = actions[-1]
                    print(f"   - 첫 프레임: {first_action}")
                    print(f"   - 마지막 프레임: {last_action}")
                    
        except Exception as e:
            print(f"❌ {h5_file} 분석 실패: {e}")
    
    # 전체 데이터셋 통계
    print(f"\n📊 전체 데이터셋 액션 통계:")
    print("=" * 50)
    
    all_actions = []
    non_zero_count = 0
    total_frames = 0
    
    for h5_file in h5_files:
        try:
            with h5py.File(h5_file, 'r') as f:
                if 'actions' in f:
                    actions = f['actions'][:]  # [18, 3]
                    all_actions.append(actions)
                    non_zero_count += np.count_nonzero(actions)
                    total_frames += actions.shape[0] * actions.shape[1]
        except Exception as e:
            print(f"❌ {h5_file} 로드 실패: {e}")
    
    if all_actions:
        all_actions = np.concatenate(all_actions, axis=0)  # [total_frames, 3]
        
        print(f"📊 전체 액션 통계:")
        print(f"   - 총 프레임 수: {all_actions.shape[0]}")
        print(f"   - 총 액션 값 수: {total_frames}")
        print(f"   - 0이 아닌 액션 값 수: {non_zero_count}")
        print(f"   - 0 비율: {(total_frames - non_zero_count) / total_frames * 100:.2f}%")
        print(f"   - 0이 아닌 비율: {non_zero_count / total_frames * 100:.2f}%")
        
        print(f"   - 전체 최솟값: [{all_actions[:, 0].min():8.6f}, {all_actions[:, 1].min():8.6f}, {all_actions[:, 2].min():8.6f}]")
        print(f"   - 전체 최댓값: [{all_actions[:, 0].max():8.6f}, {all_actions[:, 1].max():8.6f}, {all_actions[:, 2].max():8.6f}]")
        print(f"   - 전체 평균값: [{all_actions[:, 0].mean():8.6f}, {all_actions[:, 1].mean():8.6f}, {all_actions[:, 2].mean():8.6f}]")
        print(f"   - 전체 표준편차: [{all_actions[:, 0].std():8.6f}, {all_actions[:, 1].std():8.6f}, {all_actions[:, 2].std():8.6f}]")

def check_action_event_types():
    """액션 이벤트 타입 확인"""
    
    print(f"\n🔍 액션 이벤트 타입 분석")
    print("=" * 50)
    
    data_path = '../../ROS_action/mobile_vla_dataset'
    
    if os.path.isdir(data_path):
        h5_files = list(Path(data_path).glob("*.h5"))
    else:
        h5_files = [data_path]
    
    # 몇 개 파일만 확인
    sample_files = h5_files[:3]
    
    for h5_file in sample_files:
        print(f"\n🔍 {h5_file.name} 액션 이벤트 타입:")
        print("-" * 40)
        
        try:
            with h5py.File(h5_file, 'r') as f:
                if 'action_event_types' in f:
                    event_types = f['action_event_types'][:]
                    
                    print(f"📊 이벤트 타입 shape: {event_types.shape}")
                    print(f"📊 이벤트 타입 dtype: {event_types.dtype}")
                    
                    # 이벤트 타입 출력
                    for frame_idx, event_type in enumerate(event_types):
                        try:
                            # bytes를 문자열로 변환
                            if isinstance(event_type, bytes):
                                event_str = event_type.decode('utf-8')
                            else:
                                event_str = str(event_type)
                            print(f"   프레임 {frame_idx:2d}: {event_str}")
                        except:
                            print(f"   프레임 {frame_idx:2d}: {event_type}")
                
        except Exception as e:
            print(f"❌ {h5_file} 이벤트 타입 분석 실패: {e}")

if __name__ == "__main__":
    analyze_action_values()
    check_action_event_types()
