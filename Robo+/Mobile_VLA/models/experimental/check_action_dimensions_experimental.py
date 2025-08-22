#!/usr/bin/env python3
"""
액션 차원 확인 스크립트
"""
import sys
from pathlib import Path

ROOT_DIR = Path("/home/billy/25-1kp/vla/Robo+/Mobile_VLA")
DATA_DIR = Path("/home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset")
ROBOVLMS_DIR = Path("/home/billy/25-1kp/vla/RoboVLMs")

sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROBOVLMS_DIR))

from robovlms.data.mobile_vla_dataset import MobileVLADataset

def check_action_dimensions():
    print("🔍 액션 차원 확인 중...")
    
    # 데이터셋 로드
    dataset = MobileVLADataset(DATA_DIR)
    print(f"데이터셋 크기: {len(dataset)}")
    
    # 첫 번째 에피소드 확인
    episode = dataset[0]
    actions = episode['actions']
    
    print(f"액션 타입: {type(actions)}")
    print(f"액션 shape: {actions.shape}")
    print(f"액션 차원: {actions.shape[-1]}")
    
    # 모든 에피소드의 액션 차원 확인
    all_dims = []
    for i in range(min(10, len(dataset))):  # 처음 10개만 확인
        episode = dataset[i]
        actions = episode['actions']
        dim = actions.shape[-1]
        all_dims.append(dim)
        print(f"에피소드 {i}: 액션 차원 = {dim}")
    
    print(f"\n📊 액션 차원 통계:")
    print(f"   최소 차원: {min(all_dims)}")
    print(f"   최대 차원: {max(all_dims)}")
    print(f"   평균 차원: {sum(all_dims) / len(all_dims)}")
    
    # 액션 값 범위 확인
    first_actions = dataset[0]['actions']
    print(f"\n📈 첫 번째 에피소드 액션 값 범위:")
    print(f"   전체 범위: {first_actions.min():.4f} ~ {first_actions.max():.4f}")
    for i in range(first_actions.shape[-1]):
        print(f"   차원 {i}: {first_actions[:, i].min():.4f} ~ {first_actions[:, i].max():.4f}")

if __name__ == "__main__":
    check_action_dimensions()
