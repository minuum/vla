
import torch
from robovlm_nav.datasets.nav_h5_dataset_impl import MobileVLAH5Dataset
import os

def test_v4_dataset():
    data_dir = "/home/billy/25-1kp/MoNaVLA/ROS_action/mobile_vla_dataset_v3"
    print(f"Testing dataset at: {data_dir}")
    
    # V4-EXP01 설정 모사
    dataset = MobileVLAH5Dataset(
        data_dir=data_dir,
        episode_pattern="episode_*.h5",
        window_size=8,
        action_chunk_size=1,
        discrete_action=True,
        min_episode_frames=10,
        instruction_preset="center_goal",
        is_validation=False
    )
    
    print(f"Total valid samples: {len(dataset)}")
    
    # 첫 번째 샘플 로드 테스트
    sample = dataset[0]
    print("\nSample check:")
    print(f"RGB Shape: {sample['rgb'].shape}")  # Expected: (9, 3, 224, 224) -> window(8) + chunk(1)
    print(f"Actions Shape: {sample['actions'].shape}")
    print(f"Action: {sample['actions']}")
    print(f"Instruction: {sample['lang']}")
    
    # 마지막 샘플 로드 테스트 (가변 길이 대응 확인)
    last_sample = dataset[len(dataset)-1]
    print("\nLast sample check:")
    print(f"RGB Shape: {last_sample['rgb'].shape}")
    print(f"Instruction: {last_sample['lang']}")

if __name__ == "__main__":
    test_v4_dataset()
