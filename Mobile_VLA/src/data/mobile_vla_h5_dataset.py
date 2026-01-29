#!/usr/bin/env python3
"""
Mobile VLA HDF5 Dataset Implementation for 20251106 Episodes
참조: https://github.com/Robot-VLAs/RoboVLMs/blob/main/robovlms/data/calvin_dataset.py
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from PIL import Image
import cv2
import glob

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MobileVLAH5Dataset(Dataset):
    """
    Mobile VLA HDF5 데이터셋 클래스 (20251106 에피소드용)
    참조: https://github.com/Robot-VLAs/RoboVLMs/blob/main/robovlms/data/calvin_dataset.py:63-71
    CALVIN 데이터셋의 obs_config 구조를 참고하여 Mobile VLA에 맞게 수정
    
    HDF5 구조:
    - images: (T, 720, 1280, 3) uint8 - RGB 이미지
    - actions: (T, 3) float32 - [linear_x, linear_y, angular_z]
    - action_event_types: (T,) - 액션 이벤트 타입
    """
    
    def __init__(
        self,
        data_dir: str,
        episode_pattern: str = "episode_20251106_*.h5",
        window_size: int = 8,
        action_chunk_size: int = 10,
        image_size: int = 224,
        image_mean: List[float] = [0.48145466, 0.4578275, 0.40821073],
        image_std: List[float] = [0.26862954, 0.26130258, 0.27577711],
        train_split: float = 0.8,
        is_validation: bool = False,
        norm_action: bool = True,
        norm_min: float = -1.0,
        norm_max: float = 1.0
    ):
        """
        Mobile VLA HDF5 데이터셋 초기화
        
        Args:
            data_dir: 데이터 디렉토리 경로
            episode_pattern: 에피소드 파일 패턴
            window_size: 윈도우 크기 (참조: RoboVLMs window_size=8)
            action_chunk_size: 액션 청크 크기 (참조: RoboVLMs fwd_pred_next_n=10)
            image_size: 이미지 크기 (224x224)
            image_mean: 이미지 정규화 평균
            image_std: 이미지 정규화 표준편차
            train_split: 훈련 데이터 비율
            is_validation: 검증 데이터셋 여부
            norm_action: 액션 정규화 여부
            norm_min: 액션 정규화 최소값
            norm_max: 액션 정규화 최대값
        """
        self.data_dir = Path(data_dir)
        self.window_size = window_size
        self.action_chunk_size = action_chunk_size
        self.image_size = image_size
        self.image_mean = np.array(image_mean).reshape(1, 1, 3)
        self.image_std = np.array(image_std).reshape(1, 1, 3)
        self.norm_action = norm_action
        self.norm_min = norm_min
        self.norm_max = norm_max
        
        # 에피소드 파일 로드
        episode_files = sorted(glob.glob(str(self.data_dir / episode_pattern)))
        
        if not episode_files:
            raise ValueError(f"No episodes found matching pattern: {episode_pattern}")
        
        # Train/Val 분할
        split_idx = int(len(episode_files) * train_split)
        if is_validation:
            self.episode_files = episode_files[split_idx:]
            logger.info(f"📊 Validation 데이터셋: {len(self.episode_files)}개 에피소드")
        else:
            self.episode_files = episode_files[:split_idx]
            logger.info(f"📊 Training 데이터셋: {len(self.episode_files)}개 에피소드")
        
        # 샘플 인덱스 생성
        self.samples = self._build_sample_index()
        
        logger.info(f"✅ 총 {len(self.samples)}개 샘플 생성")
    
    def _build_sample_index(self) -> List[Tuple[str, int, str]]:
        """
        샘플 인덱스 생성 (에피소드 파일, 시작 프레임, 명령어)
        CSV 인덱스 파일이 있으면 그것을 사용하고, 없으면 패턴 매칭 사용
        """
        samples = []
        
        # 1. CSV 인덱스 확인
        index_csv_path = self.data_dir / "dataset_index.csv"
        instruction_map = {}
        
        if index_csv_path.exists():
            logger.info(f"📂 CSV 인덱스 파일 발견: {index_csv_path}")
            try:
                import pandas as pd
                df = pd.read_csv(index_csv_path)
                # 경로 -> 명령어 매핑 생성
                for _, row in df.iterrows():
                    instruction_map[row['episode_path']] = row['instruction']
            except Exception as e:
                logger.warning(f"CSV 로드 실패, 파일명 기반으로 진행합니다: {e}")

        # 2. 에피소드 처리
        for episode_file in self.episode_files:
            # 명령어 결정
            if episode_file in instruction_map:
                lang = instruction_map[episode_file]
            else:
                # Fallback: 파일명 기반 추론
                fname = str(Path(episode_file).name).lower()
                if "left" in fname:
                    lang = "Navigate to the brown pot on the left"
                elif "right" in fname:
                    lang = "Navigate to the brown pot on the right"
                else:
                    lang = "Navigate to the brown pot" # 기본값
            
            try:
                with h5py.File(episode_file, 'r') as f:
                    num_frames = f['images'].shape[0]
                    
                    # Window size + action chunk를 고려한 샘플 생성
                    max_start_idx = num_frames - self.window_size - self.action_chunk_size + 1
                    
                    if max_start_idx > 0:
                        for start_idx in range(max_start_idx):
                            samples.append((episode_file, start_idx, lang))
            except Exception as e:
                logger.warning(f"에피소드 로드 실패 {episode_file}: {e}")
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        이미지 전처리
        참조: https://github.com/Robot-VLAs/RoboVLMs/blob/main/robovlms/data/calvin_dataset.py:73-81
        이미지 전처리 방식 참고
        
        Args:
            image: (H, W, 3) uint8 RGB 이미지
        
        Returns:
            (3, 224, 224) float32 정규화된 이미지
        """
        # Resize to 224x224
        image = cv2.resize(image, (self.image_size, self.image_size))
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Apply ImageNet normalization
        image = (image - self.image_mean) / self.image_std
        
        # HWC -> CHW
        image = np.transpose(image, (2, 0, 1))
        
        return image
    
    def _normalize_action(self, action: np.ndarray) -> np.ndarray:
        """
        액션 정규화
        참조: https://github.com/Robot-VLAs/RoboVLMs/blob/main/robovlms/data/calvin_dataset.py
        norm_action 처리 방식 참고
        
        Args:
            action: (2,) [linear_x, linear_y] 또는 (T, 2)
        
        Returns:
            정규화된 액션 [-1, 1] 범위
        """
        if not self.norm_action:
            return action
        
        # Clip to [norm_min, norm_max]
        action = np.clip(action, self.norm_min, self.norm_max)
        
        # Normalize to [-1, 1]
        action = 2.0 * (action - self.norm_min) / (self.norm_max - self.norm_min) - 1.0
        
        return action
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        데이터 아이템 반환
        참조: https://github.com/Robot-VLAs/RoboVLMs/blob/main/robovlms/data/calvin_dataset.py:857-858
        CALVIN 데이터셋의 collater 함수 구조를 참고
        """
        episode_file, start_idx, lang = self.samples[idx]
        
        with h5py.File(episode_file, 'r') as f:
            # 이미지 로드 (window_size 프레임)
            # 참조: RoboVLMs window_size=8 사용
            images = f['images'][start_idx:start_idx + self.window_size]  # (T, H, W, 3)
            
            # 액션 로드 (action_chunk_size 프레임)
            # 참조: RoboVLMs fwd_pred_next_n=10 사용
            actions = f['actions'][start_idx + self.window_size:start_idx + self.window_size + self.action_chunk_size]  # (T, 3)
            
            # Mobile VLA는 2D 액션만 사용 (linear_x, linear_y)
            actions = actions[:, :2]  # (T, 2)
        
        # 이미지 전처리
        # 참조: https://github.com/Robot-VLAs/RoboVLMs/blob/main/robovlms/data/calvin_dataset.py:73-81
        processed_images = []
        for img in images:
            processed_img = self._preprocess_image(img)
            processed_images.append(processed_img)
        
        processed_images = np.stack(processed_images, axis=0)  # (T, 3, 224, 224)
        
        # 액션 정규화
        actions = self._normalize_action(actions)
        
        # 언어 명령 (샘플 인덱스에서 로드)
        language = lang
        
        # 액션 마스크 생성
        # 참조: https://github.com/Robot-VLAs/RoboVLMs/blob/main/robovlms/data/calvin_dataset.py:884-887
        action_mask = np.ones((self.action_chunk_size,), dtype=bool)
        
        return {
            "images": torch.from_numpy(processed_images).float(),  # (T, 3, 224, 224)
            "actions": torch.from_numpy(actions).float(),  # (T, 2)
            "action_mask": torch.from_numpy(action_mask),  # (T,)
            "language": language,
            "episode_file": episode_file
        }

def create_mobile_vla_h5_dataloader(
    data_dir: str,
    episode_pattern: str = "episode_20251106_*.h5",
    batch_size: int = 2,
    num_workers: int = 4,
    window_size: int = 8,
    action_chunk_size: int = 10,
    train_split: float = 0.8
) -> Tuple[DataLoader, DataLoader]:
    """
    Mobile VLA HDF5 데이터 로더 생성 함수
    참조: https://github.com/Robot-VLAs/RoboVLMs/blob/main/robovlms/data/calvin_dataset.py:22-25
    """
    # 훈련 데이터셋
    train_dataset = MobileVLAH5Dataset(
        data_dir=data_dir,
        episode_pattern=episode_pattern,
        window_size=window_size,
        action_chunk_size=action_chunk_size,
        train_split=train_split,
        is_validation=False
    )
    
    # 검증 데이터셋
    val_dataset = MobileVLAH5Dataset(
        data_dir=data_dir,
        episode_pattern=episode_pattern,
        window_size=window_size,
        action_chunk_size=action_chunk_size,
        train_split=train_split,
        is_validation=True
    )
    
    # DataLoader 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    logger.info(f"📊 Mobile VLA HDF5 데이터 로더 생성 완료")
    logger.info(f"  - Train: {len(train_dataset)} 샘플")
    logger.info(f"  - Val: {len(val_dataset)} 샘플")
    logger.info(f"  - Batch Size: {batch_size}")
    
    return train_loader, val_loader

def test_mobile_vla_h5_dataset():
    """Mobile VLA HDF5 데이터셋 테스트"""
    logger.info("🧪 Mobile VLA HDF5 데이터셋 테스트 시작")
    
    try:
        # 데이터 로더 생성
        train_loader, val_loader = create_mobile_vla_h5_dataloader(
            data_dir="/home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset",
            episode_pattern="episode_20251106_*.h5",
            batch_size=2,
            num_workers=0
        )
        
        # 테스트 실행
        for batch in train_loader:
            logger.info(f"✅ 배치 로드 성공:")
            logger.info(f"  - images shape: {batch['images'].shape}")
            logger.info(f"  - actions shape: {batch['actions'].shape}")
            logger.info(f"  - language: {batch['language']}")
            break
        
        logger.info("✅ Mobile VLA HDF5 데이터셋 테스트 완료!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Mobile VLA HDF5 데이터셋 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_mobile_vla_h5_dataset()

