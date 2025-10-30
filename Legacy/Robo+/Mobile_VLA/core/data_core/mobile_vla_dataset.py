# Mobile VLA Dataset for RoboVLMs Framework
# Based on RoboVLMs ActionPredictionDataset

import h5py
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# RoboVLMs ActionPredictionDataset 임포트 (실제 RoboVLMs 설치된 경우)
try:
    from robovlms.data.base_action_prediction_dataset import ActionPredictionDataset
except ImportError:
    # 대체 구현 (로컬)
    from torch.utils.data import Dataset
    
    class ActionPredictionDataset(Dataset):
        """ActionPredictionDataset 대체 구현"""
        def __init__(self, **kwargs):
            self.batch_transform = None
            self.collater_fn = None
            
        def init_batch_transform(self):
            """배치 변환 초기화 (더미)"""
            return lambda **kwargs: kwargs
            
        def init_collater_fn(self):
            """Collater 초기화 (더미)"""
            from torch.utils.data import default_collate
            return default_collate
            
        @property
        def collater(self):
            """Collater 함수"""
            if self.collater_fn is None:
                self.collater_fn = self.init_collater_fn()
            return self.collater_fn

logger = logging.getLogger(__name__)


class MobileVLADataset(ActionPredictionDataset):
    """
    Mobile VLA Dataset for Action Prediction
    
    RoboVLMs 프레임워크와 호환되는 Mobile VLA 데이터셋.
    Window/Chunk 메커니즘을 사용하여 연속적인 액션 예측을 수행.
    """
    
    def __init__(
        self,
        data_dir: str,
        model_name: str = "kosmos",
        mode: str = "train",
        organize_type: str = "segment",  # RoboVLMs 호환
        window_size: int = 8,  # 메모리 효율성을 위해 8로 축소
        fwd_pred_next_n: int = 2,
        discrete: bool = False,  # 연속 액션 공간
        norm_action: bool = True,
        norm_min: float = -1.0,
        norm_max: float = 1.0,
        use_mu_law: bool = False,
        regular_action: bool = False,
        x_mean: float = 0.0,
        x_std: float = 1.0,
        image_history: bool = True,
        action_history: bool = True,
        predict_stop_token: bool = False,
        special_history_id: int = -100,
        tokenizer: Optional[Dict] = None,
        rgb_pad: int = -1,
        gripper_pad: int = -1,
        **kwargs,
    ):
        # 시나리오 명령어 정의 (super().__init__ 전에 필요)
        self.scenario_instructions = {
            "1box_left_vertical": "Navigate around the single box obstacle by going left to track the target cup",
            "1box_left_horizontal": "Navigate around the single box obstacle by going left to track the target cup", 
            "1box_right_vertical": "Navigate around the single box obstacle by going right to track the target cup",
            "1box_right_horizontal": "Navigate around the single box obstacle by going right to track the target cup",
            "2box_left_vertical": "Navigate around the two box obstacles by going left to track the target cup",
            "2box_left_horizontal": "Navigate around the two box obstacles by going left to track the target cup",
            "2box_right_vertical": "Navigate around the two box obstacles by going right to track the target cup", 
            "2box_right_horizontal": "Navigate around the two box obstacles by going right to track the target cup",
            "unknown": "Navigate around obstacles to track the target cup",  # 기본 명령어
        }
        
        self.data_dir = Path(data_dir)
        self.window_size = window_size
        self.fwd_pred_next_n = fwd_pred_next_n
        self.mode = mode
        self.model_name = model_name
        
        # RoboVLMs 부모 클래스 초기화 (if available)
        try:
            super().__init__(
                model_name=model_name,
                mode=mode,
                organize_type=organize_type,
                discrete=discrete,
                action_history=action_history,
                image_history=image_history,
                predict_stop_token=predict_stop_token,
                special_history_id=special_history_id,
                window_size=window_size,
                fwd_pred_next_n=fwd_pred_next_n,
                norm_action=norm_action,
                norm_min=norm_min,
                norm_max=norm_max,
                regular_action=regular_action,
                x_mean=x_mean,
                x_std=x_std,
                use_mu_law=use_mu_law,
                tokenizer=tokenizer,
                rgb_pad=rgb_pad,
                gripper_pad=gripper_pad,
                **kwargs
            )
        except:
            # Fallback: 기본 Dataset 초기화만
            pass
        
        # Mobile VLA 데이터 로드
        self._load_mobile_vla_data()
        
        # Batch transform 초기화
        self.batch_transform = self._create_mobile_vla_transform()
        self.collater_fn = self._create_mobile_vla_collater()
        
    def _load_mobile_vla_data(self):
        """Mobile VLA H5 파일 로드"""
        self.episodes = []
        h5_files = list(self.data_dir.glob("*.h5"))
        
        logger.info(f"Loading Mobile VLA data from {self.data_dir}")
        logger.info(f"Found {len(h5_files)} H5 files")
        
        def _is_valid_h5(path: Path) -> bool:
            """H5 파일 유효성 검사"""
            try:
                with h5py.File(path, 'r') as f:
                    required_keys = ['images', 'actions']
                    return all(key in f for key in required_keys)
            except:
                return False
        
        valid_files = [f for f in h5_files if _is_valid_h5(f)]
        logger.info(f"Valid H5 files: {len(valid_files)}")
        
        for h5_file in valid_files:
            try:
                scenario = self._extract_scenario_from_filename(h5_file.name)
                
                with h5py.File(h5_file, 'r') as f:
                    # 데이터 로드
                    images = f['images'][:]  # [T, H, W, C]
                    actions = f['actions'][:]  # [T, 3]
                    
                    # 시퀀스 길이 확인 (충분한 길이인지)
                    if len(images) < self.window_size + self.fwd_pred_next_n:
                        logger.warning(f"Sequence too short ({len(images)} < {self.window_size + self.fwd_pred_next_n}): {h5_file.name}")
                        continue
                    
                    episode = {
                        'file_path': str(h5_file),
                        'scenario': scenario,
                        'task_description': self.scenario_instructions[scenario],
                        'images': images,
                        'actions': actions,
                        'episode_mask': np.ones(len(images), dtype=bool)  # RoboVLMs 호환
                    }
                    
                    self.episodes.append(episode)
                    
            except Exception as e:
                logger.error(f"Failed to load episode {h5_file.name}: {e}")
        
        logger.info(f"Loaded {len(self.episodes)} valid episodes")
        
        # 시나리오별 분포 출력
        scenario_counts = {}
        for ep in self.episodes:
            scenario = ep['scenario']
            scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
        
        logger.info("Scenario distribution:")
        for scenario, count in scenario_counts.items():
            logger.info(f"  {scenario}: {count} episodes")
    
    def _extract_scenario_from_filename(self, filename: str) -> str:
        """파일명에서 시나리오 추출"""
        filename_lower = filename.lower()
        
        # 더 강건한 패턴 매칭
        if "1box" in filename_lower:
            if "vert" in filename_lower or "vertical" in filename_lower:
                if "left" in filename_lower:
                    return "1box_left_vertical"
                elif "right" in filename_lower:
                    return "1box_right_vertical"
            elif "hori" in filename_lower or "horizontal" in filename_lower:
                if "left" in filename_lower:
                    return "1box_left_horizontal"
                elif "right" in filename_lower:
                    return "1box_right_horizontal"
        elif "2box" in filename_lower:
            if "vert" in filename_lower or "vertical" in filename_lower:
                if "left" in filename_lower:
                    return "2box_left_vertical"
                elif "right" in filename_lower:
                    return "2box_right_vertical"
            elif "hori" in filename_lower or "horizontal" in filename_lower:
                if "left" in filename_lower:
                    return "2box_left_horizontal"
                elif "right" in filename_lower:
                    return "2box_right_horizontal"
        
        # 시나리오를 찾지 못한 경우 unknown으로 분류하되 여전히 사용
        logger.debug(f"Unknown scenario for {filename}, using default")
        return "unknown"

    def __len__(self) -> int:
        return len(self.episodes)

    @staticmethod
    def _to_text(e: Any) -> str:
        """텍스트 변환 유틸리티"""
        if isinstance(e, str):
            return e
        elif isinstance(e, bytes):
            return e.decode()
        elif hasattr(e, 'item'):
            return str(e.item())
        else:
            return str(e)

    def _extract_scenario(self, episode_name: str) -> str:
        """에피소드명에서 시나리오 추출 (RoboVLMs 호환성)"""
        return self._extract_scenario_from_filename(episode_name)

    def _convert_to_pil_images(self, images_array):
        """numpy 배열을 PIL 이미지 리스트로 변환"""
        pil_images = []
        for i in range(images_array.shape[0]):
            img = images_array[i]
            # 0-1 범위에서 0-255로 변환 (필요한 경우)
            if img.dtype == np.float32 or img.dtype == np.float64:
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)
            pil_img = Image.fromarray(img)
            pil_images.append(pil_img)
        return pil_images

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # 안전하게 유효 파일로 접근 (드물게 런타임 손상 발생 대비)
        episode = self.episodes[idx % len(self.episodes)]
        
        task_description = episode['task_description']
        images_array = episode['images']  # [T, H, W, C]
        actions_array = episode['actions']  # [T, action_dim]
        episode_mask = episode['episode_mask']  # [T]
        
        # PIL 이미지로 변환
        images_pil = self._convert_to_pil_images(images_array)
        
        # 2D 액션만 추출 (Z축은 항상 0이므로 제거)
        actions_2d = actions_array[:, :2]  # [T, 2] - linear_x, linear_y만 사용
        
        # 간단한 배치 변환 (RoboVLMs 스타일)
        return {
            'task_description': task_description,
            'images': images_pil,
            'actions': actions_2d,  # 2D 액션 사용
            'episode_mask': episode_mask,
            'scenario': episode['scenario']
        }
    
    def _create_mobile_vla_transform(self):
        """Mobile VLA용 batch transform 생성"""
        def transform_fn(**kwargs):
            # 단순히 데이터를 그대로 반환
            return kwargs
        return transform_fn
    
    def _create_mobile_vla_collater(self):
        """Mobile VLA용 collater 생성"""
        def collate_fn(batch):
            """배치 데이터 collation"""
            from transformers import AutoProcessor
            
            # 배치 크기
            batch_size = len(batch)
            
            # 데이터 분리
            task_descriptions = [item['task_description'] for item in batch]
            images_list = [item['images'] for item in batch]  # List[List[PIL.Image]]
            actions_list = [item['actions'] for item in batch]  # List[np.ndarray]
            episode_masks = [item['episode_mask'] for item in batch]
            scenarios = [item['scenario'] for item in batch]
            
            # 이미지 처리 (Kosmos 호환)
            try:
                processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
                
                # 모든 이미지를 하나의 리스트로 flatten
                all_images = []
                for images in images_list:
                    all_images.extend(images)
                
                # 배치 처리
                if all_images:
                    processed = processor(images=all_images, return_tensors="pt")
                    pixel_values = processed["pixel_values"]
                    
                    # 배치별로 다시 분할
                    images_per_batch = [len(images) for images in images_list]
                    pixel_values_list = []
                    start_idx = 0
                    for count in images_per_batch:
                        pixel_values_list.append(pixel_values[start_idx:start_idx + count])
                        start_idx += count
                else:
                    pixel_values_list = [torch.empty(0, 3, 224, 224) for _ in range(batch_size)]
            except:
                # Fallback: 빈 텐서
                pixel_values_list = [torch.empty(0, 3, 224, 224) for _ in range(batch_size)]
            
            # 액션 텐서 변환
            actions_tensors = []
            for actions in actions_list:
                actions_tensor = torch.from_numpy(actions).float()
                actions_tensors.append(actions_tensor)
            
            # Window/Chunk 처리 (간단버전)
            window_size = 8
            chunk_size = 2
            
            action_chunks = []
            for actions in actions_tensors:
                if len(actions) >= window_size + chunk_size:
                    # 마지막 chunk_size 프레임을 타겟으로 사용
                    chunk = actions[-chunk_size:]
                    action_chunks.append(chunk)
                else:
                    # 패딩 또는 반복
                    chunk = actions[-chunk_size:] if len(actions) >= chunk_size else actions
                    action_chunks.append(chunk)
            
            if action_chunks:
                action_chunck = torch.stack(action_chunks)  # [B, chunk_size, action_dim]
            else:
                action_chunck = torch.empty(batch_size, chunk_size, 3)
            
            return {
                'images': torch.stack(pixel_values_list) if pixel_values_list else torch.empty(batch_size, 0, 3, 224, 224),  # trainer 호환
                'actions': torch.stack(actions_tensors) if actions_tensors else torch.empty(batch_size, 0, 3),
                'target_actions': action_chunck,  # 예측 타겟
                'task_description': task_descriptions,
                'scenarios': scenarios,
                'data_source': 'mobile_vla'
            }
        
        return collate_fn
    
    @property
    def collater(self):
        """Collater 함수 반환"""
        if self.collater_fn is None:
            self.collater_fn = self._create_mobile_vla_collater()
        return self.collater_fn
