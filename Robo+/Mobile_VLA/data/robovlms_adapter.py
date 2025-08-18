#!/usr/bin/env python3
"""
RoboVLMs Adapter for Mobile VLA - Mobile VLA를 RoboVLMs와 호환시키는 어댑터
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path

try:
    from .mobile_dataset import MobileVLADataset
except ImportError:
    # 테스트용 절대 임포트
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from mobile_dataset import MobileVLADataset


class MobileVLAToRoboVLMsAdapter:
    """
    Mobile VLA 데이터를 RoboVLMs ActionPredictionDataset 형식으로 변환하는 어댑터
    """
    
    def __init__(
        self,
        data_dir: str,
        sequence_length: int = 18,
        scenario_filter: Optional[List[str]] = None,
        image_processor=None  # Kosmos processor
    ):
        # Mobile VLA 데이터셋 로드
        self.mobile_dataset = MobileVLADataset(
            data_dir=data_dir,
            sequence_length=sequence_length,
            scenario_filter=scenario_filter
        )
        
        self.image_processor = image_processor
        self.sequence_length = sequence_length
        
        # RoboVLMs 시나리오 명령어 매핑 (영어 - 장애물 회피 컵 추적)
        self.scenario_instructions = {
            "1box_vert_left": "Navigate around the single box obstacle by going left to track the target cup",
            "1box_vert_right": "Navigate around the single box obstacle by going right to track the target cup",
            "1box_hori_left": "Navigate around the single box obstacle by going left to track the target cup",
            "1box_hori_right": "Navigate around the single box obstacle by going right to track the target cup",
            "2box_vert_left": "Navigate around the two box obstacles by going left to track the target cup",
            "2box_vert_right": "Navigate around the two box obstacles by going right to track the target cup",
            "2box_hori_left": "Navigate around the two box obstacles by going left to track the target cup",
            "2box_hori_right": "Navigate around the two box obstacles by going right to track the target cup"
        }
    
    def __len__(self) -> int:
        return len(self.mobile_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """RoboVLMs ActionPredictionDataset 형식으로 변환"""
        # Mobile VLA 데이터 로드
        mobile_data = self.mobile_dataset[idx]
        
        # 1. 이미지 처리 (Kosmos processor 사용)
        images = mobile_data["images"]  # [T, 3, 224, 224]
        
        if self.image_processor is not None:
            # Kosmos processor 사용
            # PIL Images로 변환
            from PIL import Image
            pil_images = []
            for t in range(images.shape[0]):
                # tensor [3, 224, 224] → numpy [224, 224, 3] → PIL
                img_np = images[t].permute(1, 2, 0).numpy()
                img_np = (img_np * 255).astype(np.uint8)  # denormalize
                pil_img = Image.fromarray(img_np)
                pil_images.append(pil_img)
            
            # Kosmos processor로 처리
            processed_images = []
            for pil_img in pil_images:
                processed = self.image_processor(pil_img, return_tensors="pt")['pixel_values']
                processed_images.append(processed.squeeze(0))  # [3, 224, 224]
            
            vision_x = torch.stack(processed_images)  # [T, 3, 224, 224]
        else:
            # 기본 처리
            vision_x = images
        
        # 2. 액션 그대로 사용 (Mobile VLA 3D 유지)
        mobile_actions = mobile_data["actions"]  # [T, 3] normalized - 그대로 사용
        
        # 3. 이벤트 그대로 사용 (Mobile VLA 이벤트 유지)
        event_indices = mobile_data["action_events"]  # [T]
        
        # 4. 시나리오 명령어
        scenario = mobile_data["scenario"]
        task_description = self.scenario_instructions.get(scenario, "Navigate to track the target cup")
        
        # Mobile VLA 형식으로 반환 (RoboVLMs 방식만 차용)
        return {
            # 이미지 데이터
            "vision_x": vision_x.unsqueeze(0),  # [1, T, 3, 224, 224] - batch 차원 추가
            
            # 태스크 설명
            "task_description": task_description,
            
            # 메타데이터
            "scenario": scenario,
            "episode_name": mobile_data["episode_name"],
            "num_frames": mobile_data["num_frames"],
            
            # Mobile VLA 원본 데이터 (학습용)
            "mobile_actions": mobile_data["actions"].unsqueeze(0),       # [1, T, 3] - Mobile VLA 액션
            "mobile_events": mobile_data["action_events"].unsqueeze(0)   # [1, T] - Mobile VLA 이벤트
        }
    
    # 변환 함수들은 필요없음 - Mobile VLA 데이터 그대로 사용
    
    def get_scenario_statistics(self) -> Dict[str, int]:
        """시나리오별 통계 반환"""
        return self.mobile_dataset.get_scenario_statistics()


def create_robovlms_compatible_dataloader(
    data_dir: str,
    batch_size: int = 1,
    sequence_length: int = 18,
    scenario_filter: Optional[List[str]] = None,
    image_processor=None,
    num_workers: int = 0
):
    """RoboVLMs 호환 DataLoader 생성"""
    from torch.utils.data import DataLoader
    
    adapter = MobileVLAToRoboVLMsAdapter(
        data_dir=data_dir,
        sequence_length=sequence_length,
        scenario_filter=scenario_filter,
        image_processor=image_processor
    )
    
    def collate_fn(batch):
        """배치 처리"""
        # 단일 배치라고 가정 (RoboVLMs는 보통 batch_size=1)
        if len(batch) == 1:
            return batch[0]
        
        # 여러 개인 경우 첫 번째만 반환 (간단히)
        return batch[0]
    
    return DataLoader(
        adapter,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )


if __name__ == "__main__":
    # 테스트 코드
    print("🧪 RoboVLMs Adapter 테스트")
    
    # 어댑터 초기화
    adapter = MobileVLAToRoboVLMsAdapter(
        data_dir="/home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset/"
    )
    
    if len(adapter) > 0:
        # 첫 번째 샘플 테스트
        sample = adapter[0]
        
        print(f"📊 Mobile VLA + Kosmos 방식 데이터:")
        print(f"   Vision X: {sample['vision_x'].shape}")
        print(f"   Mobile Actions: {sample['mobile_actions'].shape}")
        print(f"   Mobile Events: {sample['mobile_events'].shape}")
        print(f"   Task: {sample['task_description']}")
        print(f"   Scenario: {sample['scenario']}")
        
        # Mobile VLA 원본 데이터 확인
        print(f"\n🤖 Mobile VLA 데이터:")
        print(f"   Actions (3D): {sample['mobile_actions'][0, :3]}")  # 처음 3프레임
        print(f"   Events: {sample['mobile_events'][0, :10]}")        # 처음 10프레임
    
    print(f"\n✅ RoboVLMs Adapter 테스트 완료!")
