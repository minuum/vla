#!/usr/bin/env python3
"""
Mobile VLA Dataset - mobile_vla_data_collector.py 데이터 직접 로딩
Calvin 없이 순수 Mobile HDF5 형식 사용
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MobileVLADataset(Dataset):
    """
    mobile_vla_data_collector.py가 생성한 HDF5 파일을 직접 로딩하는 데이터셋
    
    데이터 형식:
    - images: [T, 720, 1280, 3] - RGB 이미지 시퀀스
    - actions: [T, 3] - [linear_x, linear_y, angular_z] 
    - action_event_types: [T] - [episode_start, start_action, stop_action]
    - episode_name: str - "episode_20250808_123136_1box_vert_left"
    """
    
    def __init__(
        self, 
        data_dir: str = "/home/soda/vla/ROS_action/mobile_vla_dataset/",
        sequence_length: int = 18,
        image_size: Tuple[int, int] = (224, 224),  # VLM 입력용 리사이즈
        normalize_actions: bool = True,
        scenario_filter: Optional[List[str]] = None
    ):
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.normalize_actions = normalize_actions
        
        # mobile_vla_data_collector.py의 시나리오 매핑 (장애물 개수/배치와 무관하게 최좌/최우 외곽 경로)
        self.scenario_instructions = {
            "1box_vert_left": "가장 왼쪽 외곽으로 돌아 컵까지 가세요",
            "1box_vert_right": "가장 오른쪽 외곽으로 돌아 컵까지 가세요", 
            "1box_hori_left": "가장 왼쪽 외곽으로 돌아 컵까지 가세요",
            "1box_hori_right": "가장 오른쪽 외곽으로 돌아 컵까지 가세요",
            "2box_vert_left": "가장 왼쪽 외곽으로 돌아 컵까지 가세요",
            "2box_vert_right": "가장 오른쪽 외곽으로 돌아 컵까지 가세요",
            "2box_hori_left": "가장 왼쪽 외곽으로 돌아 컵까지 가세요", 
            "2box_hori_right": "가장 오른쪽 외곽으로 돌아 컵까지 가세요"
        }
        
        # mobile_vla_data_collector.py의 액션 범위 (WASD_TO_CONTINUOUS 기준)
        self.action_bounds = {
            "linear_x": 2.0,   # 실제로는 ±1.15 사용하지만 여유있게
            "linear_y": 2.0,   # 실제로는 ±1.15 사용하지만 여유있게  
            "angular_z": 2.0   # 실제로는 ±1.15 사용하지만 여유있게
        }
        
        # 이벤트 타입 매핑
        # 문자열/바이트 문자열 모두 안전하게 처리
        self.event_mapping = {
            'episode_start': 0,
            'start_action': 1,
            'stop_action': 2
        }
        
        # HDF5 파일 로드 및 필터링
        self.h5_files = self._load_h5_files(scenario_filter)
        self.scenarios = self._extract_scenarios()
        
        # 데이터셋 통계 출력
        self._print_dataset_stats()
        
    def _load_h5_files(self, scenario_filter: Optional[List[str]]) -> List[Path]:
        """mobile_vla_data_collector.py 형태의 HDF5 파일들만 로드하고 필터링"""
        all_h5_files = list(self.data_dir.glob("*.h5"))
        
        # mobile_vla_data_collector.py 형태만 필터링
        valid_files = []
        for h5_file in all_h5_files:
            if self._is_valid_mobile_vla_format(h5_file):
                scenario = self._extract_scenario_from_filename(h5_file.name)
                
                # unknown 시나리오 제외 (태그가 없는 파일들)
                if scenario == "unknown":
                    logger.warning(f"파일 제외 (시나리오 태그 없음): {h5_file.name}")
                    continue
                
                if scenario_filter:
                    if scenario in scenario_filter:
                        valid_files.append(h5_file)
                else:
                    valid_files.append(h5_file)
            else:
                logger.warning(f"파일 제외 (mobile_vla_data_collector 형식 아님): {h5_file.name}")
        
        return valid_files
    
    def _is_valid_mobile_vla_format(self, h5_path: Path) -> bool:
        """mobile_vla_data_collector.py 형태인지 검증"""
        try:
            if not h5py.is_hdf5(h5_path):
                return False
                
            with h5py.File(h5_path, 'r') as f:
                # 필수 키 검사
                required_keys = ['images', 'actions', 'action_event_types']
                if not all(key in f for key in required_keys):
                    return False
                
                # 필수 속성 검사 
                required_attrs = ['episode_name', 'num_frames']
                if not all(attr in f.attrs for attr in required_attrs):
                    return False
                
                # 데이터 형태 검사
                images = f['images']
                actions = f['actions'] 
                events = f['action_event_types']
                
                # 차원 검사: images [T, H, W, 3], actions [T, 3], events [T]
                if len(images.shape) != 4 or images.shape[3] != 3:
                    return False
                if len(actions.shape) != 2 or actions.shape[1] != 3:
                    return False
                if len(events.shape) != 1:
                    return False
                    
                # 길이 일치 검사
                T = images.shape[0]
                if actions.shape[0] != T or events.shape[0] != T:
                    return False
                
                # 이벤트 타입이 올바른 문자열인지 검사
                sample_event = events[0]
                if isinstance(sample_event, bytes):
                    sample_event = sample_event.decode('utf-8')
                valid_events = ['episode_start', 'start_action', 'stop_action']
                if sample_event not in valid_events:
                    return False
                
                return True
                
        except Exception:
            return False
    
    def _extract_scenario_from_filename(self, filename: str) -> str:
        """파일명에서 시나리오 추출 (mobile_vla_data_collector.py 방식)"""
        for scenario in self.scenario_instructions.keys():
            if scenario in filename:
                return scenario
        return "unknown"
    
    def _extract_scenarios(self) -> List[str]:
        """모든 파일의 시나리오 추출"""
        scenarios = []
        for h5_file in self.h5_files:
            scenario = self._extract_scenario_from_filename(h5_file.name)
            scenarios.append(scenario)
        return scenarios
    
    def _print_dataset_stats(self):
        """데이터셋 통계 출력"""
        scenario_counts = defaultdict(int)
        total_frames = 0
        
        for i, h5_file in enumerate(self.h5_files):
            scenario = self.scenarios[i]
            scenario_counts[scenario] += 1
            
            # 프레임 수 확인
            try:
                with h5py.File(h5_file, 'r') as f:
                    num_frames = f.attrs.get('num_frames', 0)
                    total_frames += num_frames
            except Exception as e:
                logger.warning(f"파일 읽기 실패 {h5_file.name}: {e}")
        
        logger.info(f"📁 Mobile VLA Dataset 로드 완료!")
        logger.info(f"📊 총 {len(self.h5_files)}개 에피소드, {total_frames}개 프레임")
        logger.info(f"🎯 시나리오 분포: {dict(scenario_counts)}")
        
        # 18프레임 에피소드 특별 표시
        frame_18_count = sum(1 for scenario in scenario_counts.keys() if scenario != "unknown")
        logger.info(f"🎯 18프레임 에피소드: {frame_18_count}개 (표준 길이)")
    
    def __len__(self) -> int:
        return len(self.h5_files)
    
    def __getitem__(self, idx: int) -> Dict:
        """단일 에피소드 데이터 로드"""
        h5_file = self.h5_files[idx]
        scenario = self.scenarios[idx]
        
        try:
            with h5py.File(h5_file, 'r') as f:
                # mobile_vla_data_collector.py 데이터 직접 로드
                images = f['images'][:]                    # [T, 720, 1280, 3]
                actions = f['actions'][:]                  # [T, 3] 
                action_events = f['action_event_types'][:]  # [T]
                
                # 메타데이터
                episode_name = f.attrs['episode_name']
                num_frames = f.attrs['num_frames']
                duration = f.attrs['total_duration']
                
        except Exception as e:
            logger.error(f"HDF5 파일 로드 실패 {h5_file.name}: {e}")
            # 빈 데이터 반환
            return self._get_empty_sample(scenario)
        
        # 데이터 전처리
        processed_data = self._preprocess_episode(
            images, actions, action_events, scenario, episode_name, num_frames, duration
        )
        
        return processed_data
    
    def _preprocess_episode(
        self, 
        images: np.ndarray, 
        actions: np.ndarray, 
        action_events: np.ndarray,
        scenario: str,
        episode_name: str,
        num_frames: int,
        duration: float
    ) -> Dict:
        """에피소드 데이터 전처리"""
        
        # 1. 이미지 전처리 (720p → 224x224 리사이즈 + 정규화)
        processed_images = self._preprocess_images(images)  # [T, 3, 224, 224]
        
        # 2. 액션 정규화 (mobile_vla_data_collector.py 기준)
        if self.normalize_actions:
            processed_actions = self._normalize_actions(actions)  # [T, 3] normalized
        else:
            processed_actions = torch.FloatTensor(actions)
        
        # 3. 이벤트 타입 변환
        # h5py가 반환하는 형식이 str 또는 bytes(np.bytes_)일 수 있어 통합 처리
        def _to_text(e):
            if isinstance(e, bytes):
                return e.decode('utf-8', errors='ignore')
            try:
                import numpy as _np
                if isinstance(e, _np.bytes_):
                    return e.decode('utf-8', errors='ignore')
            except Exception:
                pass
            return str(e)

        event_indices = np.array([
            self.event_mapping.get(_to_text(event), 1) for event in action_events
        ])
        processed_events = torch.LongTensor(event_indices)  # [T]
        
        # 4. 시퀀스 길이 맞추기 (18프레임 표준)
        if len(processed_images) != self.sequence_length:
            processed_images, processed_actions, processed_events = self._pad_or_truncate_sequence(
                processed_images, processed_actions, processed_events
            )
        
        # 5. 한국어 명령어 추가
        instruction = self.scenario_instructions.get(scenario, "컵까지 가세요")
        
        return {
            "images": processed_images,              # [18, 3, 224, 224]
            "actions": processed_actions,            # [18, 3]
            "action_events": processed_events,       # [18]
            "scenario": scenario,                    # str
            "instruction": instruction,              # str (한국어)
            "episode_name": episode_name,            # str
            "num_frames": num_frames,                # int
            "duration": duration,                    # float
            "sequence_mask": torch.ones(self.sequence_length, dtype=torch.bool)  # [18] - 모든 프레임 유효
        }
    
    def _preprocess_images(self, images: np.ndarray) -> torch.Tensor:
        """이미지 전처리: 720p → 224x224 리사이즈 + 정규화"""
        import torchvision.transforms as transforms
        
        # [T, 720, 1280, 3] → [T, 3, 224, 224]
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.image_size),
            transforms.ToTensor(),  # [0, 1] 정규화 + HWC→CHW
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet 정규화
        ])
        
        processed_images = []
        for i in range(len(images)):
            # uint8 [720, 1280, 3] → normalized [3, 224, 224]
            img_tensor = transform(images[i])
            processed_images.append(img_tensor)
        
        return torch.stack(processed_images)  # [T, 3, 224, 224]
    
    def _normalize_actions(self, actions: np.ndarray) -> torch.Tensor:
        """액션 정규화 (mobile_vla_data_collector.py 기준)"""
        # [T, 3] actions: [linear_x, linear_y, angular_z]
        normalized_actions = actions.copy()
        
        # 각 축별로 [-1, 1] 범위로 정규화
        normalized_actions[:, 0] = actions[:, 0] / self.action_bounds["linear_x"]    # linear_x
        normalized_actions[:, 1] = actions[:, 1] / self.action_bounds["linear_y"]    # linear_y  
        normalized_actions[:, 2] = actions[:, 2] / self.action_bounds["angular_z"]   # angular_z
        
        # 클램핑 [-1, 1]
        normalized_actions = np.clip(normalized_actions, -1.0, 1.0)
        
        return torch.FloatTensor(normalized_actions)
    
    def _pad_or_truncate_sequence(
        self, 
        images: torch.Tensor, 
        actions: torch.Tensor, 
        events: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """시퀀스 길이를 표준 길이(18)에 맞추기"""
        current_length = len(images)
        
        if current_length == self.sequence_length:
            return images, actions, events
        elif current_length < self.sequence_length:
            # 패딩: 마지막 프레임 반복
            pad_length = self.sequence_length - current_length
            
            # 이미지 패딩
            last_image = images[-1:].repeat(pad_length, 1, 1, 1)
            padded_images = torch.cat([images, last_image], dim=0)
            
            # 액션 패딩 (정지 액션으로)
            stop_action = torch.zeros(pad_length, 3)
            padded_actions = torch.cat([actions, stop_action], dim=0)
            
            # 이벤트 패딩 (stop_action으로)
            stop_events = torch.full((pad_length,), 2, dtype=torch.long)  # stop_action = 2
            padded_events = torch.cat([events, stop_events], dim=0)
            
            return padded_images, padded_actions, padded_events
        else:
            # 자르기: 처음 sequence_length만 사용
            return images[:self.sequence_length], actions[:self.sequence_length], events[:self.sequence_length]
    
    def _get_empty_sample(self, scenario: str) -> Dict:
        """빈 샘플 반환 (에러 발생시)"""
        return {
            "images": torch.zeros(self.sequence_length, 3, *self.image_size),
            "actions": torch.zeros(self.sequence_length, 3),
            "action_events": torch.zeros(self.sequence_length, dtype=torch.long),
            "scenario": scenario,
            "instruction": self.scenario_instructions.get(scenario, "컵까지 가세요"),
            "episode_name": "error_episode",
            "num_frames": 0,
            "duration": 0.0,
            "sequence_mask": torch.zeros(self.sequence_length, dtype=torch.bool)
        }
    
    def denormalize_actions(self, normalized_actions: torch.Tensor) -> torch.Tensor:
        """정규화된 액션을 원래 범위로 복원"""
        # [-1, 1] → mobile_vla_data_collector.py 범위
        denormalized = normalized_actions.clone()
        denormalized[:, 0] *= self.action_bounds["linear_x"]    # linear_x
        denormalized[:, 1] *= self.action_bounds["linear_y"]    # linear_y
        denormalized[:, 2] *= self.action_bounds["angular_z"]   # angular_z
        return denormalized
    
    def get_scenario_statistics(self) -> Dict[str, int]:
        """시나리오별 통계 반환"""
        scenario_counts = defaultdict(int)
        for scenario in self.scenarios:
            scenario_counts[scenario] += 1
        return dict(scenario_counts)


if __name__ == "__main__":
    # 테스트 코드
    print("🧪 Mobile VLA Dataset 테스트")
    
    dataset = MobileVLADataset()
    print(f"📊 데이터셋 크기: {len(dataset)}")
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"🖼️ 이미지 형태: {sample['images'].shape}")
        print(f"🎮 액션 형태: {sample['actions'].shape}")
        print(f"⚡ 이벤트 형태: {sample['action_events'].shape}")
        print(f"🎯 시나리오: {sample['scenario']}")
        print(f"🗣️ 명령어: {sample['instruction']}")
        print(f"📋 에피소드명: {sample['episode_name']}")
    
    # 시나리오 통계
    stats = dataset.get_scenario_statistics()
    print(f"📈 시나리오 통계: {stats}")
