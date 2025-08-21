#!/usr/bin/env python3
"""
🔍 데이터셋 구조 디버깅
"""

import torch
import numpy as np
from pathlib import Path
import sys

# 프로젝트 경로 설정
ROOT_DIR = Path("/home/billy/25-1kp/vla/Robo+/Mobile_VLA")
DATA_DIR = Path("/home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset")
ROBOVLMS_DIR = Path("/home/billy/25-1kp/vla/RoboVLMs")

sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROBOVLMS_DIR))

from robovlms.data.mobile_vla_dataset import MobileVLADataset

def debug_dataset_structure():
    """데이터셋 구조 상세 분석"""
    print("🔍 데이터셋 구조 디버깅 시작")
    print("=" * 50)
    
    # 데이터셋 로드
    dataset = MobileVLADataset(DATA_DIR)
    print(f"데이터셋 크기: {len(dataset)}")
    
    # 첫 번째 에피소드 분석
    print("\n📊 첫 번째 에피소드 분석:")
    episode = dataset[0]
    
    print(f"에피소드 키: {list(episode.keys())}")
    
    # 액션 분석
    actions = episode['actions']
    print(f"\n🎮 액션 분석:")
    print(f"   타입: {type(actions)}")
    print(f"   길이: {len(actions) if hasattr(actions, '__len__') else 'N/A'}")
    
    if isinstance(actions, list):
        print(f"   첫 번째 액션: {actions[0]}")
        print(f"   첫 번째 액션 타입: {type(actions[0])}")
        print(f"   첫 번째 액션 길이: {len(actions[0]) if hasattr(actions[0], '__len__') else 'N/A'}")
        
        # numpy로 변환 시도
        try:
            actions_np = np.array(actions)
            print(f"   numpy 변환 성공: {actions_np.shape}")
            print(f"   numpy 타입: {actions_np.dtype}")
        except Exception as e:
            print(f"   numpy 변환 실패: {e}")
        
        # torch로 변환 시도
        try:
            actions_torch = torch.tensor(actions)
            print(f"   torch 변환 성공: {actions_torch.shape}")
            print(f"   torch 타입: {actions_torch.dtype}")
        except Exception as e:
            print(f"   torch 변환 실패: {e}")
    
    elif isinstance(actions, np.ndarray):
        print(f"   shape: {actions.shape}")
        print(f"   dtype: {actions.dtype}")
        print(f"   첫 번째 액션: {actions[0]}")
    
    elif isinstance(actions, torch.Tensor):
        print(f"   shape: {actions.shape}")
        print(f"   dtype: {actions.dtype}")
        print(f"   첫 번째 액션: {actions[0]}")
    
    # 이미지 분석
    images = episode['images']
    print(f"\n🖼️ 이미지 분석:")
    print(f"   타입: {type(images)}")
    print(f"   길이: {len(images) if hasattr(images, '__len__') else 'N/A'}")
    
    if isinstance(images, list):
        print(f"   첫 번째 이미지 타입: {type(images[0])}")
        if hasattr(images[0], 'shape'):
            print(f"   첫 번째 이미지 shape: {images[0].shape}")
    
    # 여러 에피소드 확인
    print(f"\n🔍 여러 에피소드 액션 타입 확인:")
    for i in range(min(5, len(dataset))):
        episode = dataset[i]
        actions = episode['actions']
        print(f"   에피소드 {i}: {type(actions)}")
        
        if isinstance(actions, list):
            print(f"     길이: {len(actions)}")
            if len(actions) > 0:
                print(f"     첫 번째 액션 타입: {type(actions[0])}")
                if hasattr(actions[0], '__len__'):
                    print(f"     첫 번째 액션 길이: {len(actions[0])}")

def test_action_processing():
    """액션 처리 테스트"""
    print("\n🧪 액션 처리 테스트")
    print("=" * 30)
    
    dataset = MobileVLADataset(DATA_DIR)
    episode = dataset[0]
    actions = episode['actions']
    
    print(f"원본 액션 타입: {type(actions)}")
    
    # 다양한 변환 방법 시도
    methods = [
        ("torch.tensor(actions)", lambda: torch.tensor(actions)),
        ("np.array(actions)", lambda: np.array(actions)),
        ("torch.tensor(actions, dtype=torch.float32)", lambda: torch.tensor(actions, dtype=torch.float32)),
        ("np.array(actions, dtype=np.float32)", lambda: np.array(actions, dtype=np.float32)),
    ]
    
    for name, method in methods:
        try:
            result = method()
            print(f"✅ {name}: 성공 - {type(result)}, shape: {getattr(result, 'shape', 'N/A')}")
        except Exception as e:
            print(f"❌ {name}: 실패 - {e}")

if __name__ == "__main__":
    debug_dataset_structure()
    test_action_processing()
