"""
Mobile VLA Dataset Adapter for RoboVLMs Framework
Adapts Mobile VLA dataset to RoboVLMs ActionPredictionDataset interface
"""

import h5py
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
import json

from robovlms.data.base_action_prediction_dataset import ActionPredictionDataset

logger = logging.getLogger(__name__)


class MobileVLADataset(ActionPredictionDataset):
    """
    Mobile VLA Dataset for RoboVLMs Framework
    
    Loads .h5 files from Mobile VLA dataset and converts to RoboVLMs format.
    Supports window/chunk mechanism for sequential action prediction.
    
    Action Space: 4D continuous
        - linear_x: forward/backward velocity (m/s)
        - linear_y: left/right velocity (m/s)  
        - angular_z: rotation velocity (rad/s)
        - action_type: discrete action type (0-3)
    """
    
    def __init__(
        self,
        data_dir: str,
        model_name: str = "kosmos",
        mode: str = "train",
        organize_type: str = "segment",
        window_size: int = 8,
        fwd_pred_next_n: int = 1,
        discrete: bool = False,
        norm_action: bool = True,
        norm_min: float = -1.0,
        norm_max: float = 1.0,
        image_size: int = 224,
        split: str = "train",
        val_ratio: float = 0.1,
        **kwargs,
    ):
        """
        Args:
            data_dir: Path to Mobile VLA dataset directory
            model_name: VLM model name (kosmos, paligemma, etc.)
            mode: train or val
            organize_type: segment (RoboVLMs standard)
            window_size: History length for observations
            fwd_pred_next_n: Number of future actions to predict
            discrete: Use discrete action space (False for continuous)
            norm_action: Normalize actions to [-1, 1]
            image_size: Image size for VLM input
            split: train or val
            val_ratio: Validation split ratio
        """
        self.data_dir = Path(data_dir)
        self.window_size = window_size
        self.fwd_pred_next_n = fwd_pred_next_n
        self.mode = mode
        self.model_name = model_name
        self.image_size = image_size
        self.norm_action = norm_action
        self.norm_min = norm_min
        self.norm_max = norm_max
        self.split = split
        self.val_ratio = val_ratio
        
        # Action bounds for Mobile VLA (2D action space)
        self.action_bounds = {
            'linear_x': (-2.0, 2.0),
            'linear_y': (-1.15, 1.15)
        }
        
        # Scenario instructions for language conditioning (Mobile VLA style)
        self.scenario_instructions = {
            "1box_vert_left": "Navigate around the single box obstacle by going left",
            "1box_vert_right": "Navigate around the single box obstacle by going right",
            "1box_hori_left": "Navigate around the single box obstacle by going left",
            "1box_hori_right": "Navigate around the single box obstacle by going right",
            "2box_vert_left": "Navigate around two box obstacles by going left",
            "2box_vert_right": "Navigate around two box obstacles by going right",
            "2box_hori_left": "Navigate around two box obstacles by going left",
            "2box_hori_right": "Navigate around two box obstacles by going right",
            "unknown": "Navigate around obstacles",
        }
        
        # Create image preprocessing function
        def image_fn(image):
            """Preprocess image for VLM"""
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image.astype(np.uint8))
            # Resize to target size
            image = image.resize((image_size, image_size))
            return image
        
        # Create tokenizer (will be loaded by parent class)
        from transformers import AutoProcessor
        tokenizer = AutoProcessor.from_pretrained(
            "microsoft/kosmos-2-patch14-224"
        )
        
        # Initialize parent class
        super().__init__(
            model_name=model_name,
            mode=mode,
            organize_type=organize_type,
            window_size=window_size,
            fwd_pred_next_n=fwd_pred_next_n,
            discrete=discrete,
            norm_action=norm_action,
            norm_min=norm_min,
            norm_max=norm_max,
            image_fn=image_fn,
            tokenizer=tokenizer,
            **kwargs
        )
        
        # Load dataset
        self.episodes = self._load_episodes()
        self.samples = self._prepare_samples()
        
        logger.info(f"Loaded {len(self.episodes)} episodes, {len(self.samples)} samples")
    
    def _load_episodes(self) -> List[Dict]:
        """Load all .h5 episode files"""
        episodes = []
        h5_files = sorted(self.data_dir.glob("*.h5"))
        
        if not h5_files:
            logger.warning(f"No .h5 files found in {self.data_dir}")
            return episodes
        
        # Split train/val
        n_val = int(len(h5_files) * self.val_ratio)
        if self.split == "val":
            h5_files = h5_files[:n_val]
        else:
            h5_files = h5_files[n_val:]
        
        for h5_path in h5_files:
            try:
                with h5py.File(h5_path, 'r') as f:
                    # Extract scenario from filename
                    scenario = self._extract_scenario(h5_path.stem)
                    
                    episode = {
                        'file_path': str(h5_path),
                        'scenario': scenario,
                        'instruction': self.scenario_instructions.get(scenario, self.scenario_instructions['unknown']),
                        'length': len(f['observations/images']),
                    }
                    episodes.append(episode)
            except Exception as e:
                logger.warning(f"Failed to load {h5_path}: {e}")
        
        return episodes
    
    def _extract_scenario(self, filename: str) -> str:
        """Extract scenario from filename"""
        # Format: episode_YYYYMMDD_HHMMSS_SCENARIO_PATTERN_DISTANCE
        parts = filename.split('_')
        if len(parts) >= 5:
            # e.g., 1box_vert_left
            scenario = '_'.join(parts[3:6])
            return scenario
        return "unknown"
    
    def _prepare_samples(self) -> List[Dict]:
        """
        Prepare samples with sliding window
        Each sample contains window_size history + fwd_pred_next_n future
        """
        samples = []
        
        for ep_idx, episode in enumerate(self.episodes):
            ep_len = episode['length']
            
            # Need at least window_size + fwd_pred_next_n frames
            if ep_len < self.window_size + self.fwd_pred_next_n:
                continue
            
            # Sliding window
            for start_idx in range(ep_len - self.window_size - self.fwd_pred_next_n + 1):
                sample = {
                    'episode_idx': ep_idx,
                    'start_idx': start_idx,
                    'end_idx': start_idx + self.window_size + self.fwd_pred_next_n,
                }
                samples.append(sample)
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample
        
        Returns:
            dict with keys:
                - images: (window_size, H, W, 3) or (window_size, 3, H, W)
                - actions: (window_size + fwd_pred_next_n - 1, 4)
                - instruction: str
                - robot_obs: (window_size, obs_dim) - optional
        """
        sample = self.samples[idx]
        episode = self.episodes[sample['episode_idx']]
        
        # Load episode data
        with h5py.File(episode['file_path'], 'r') as f:
            # Load images
            start = sample['start_idx']
            end = sample['end_idx']
            
            images = f['observations/images'][start:end]  # (T, H, W, 3)
            actions = f['action'][start:end]  # (T, 4)
            
            # Convert to torch tensors
            images = torch.from_numpy(images).float()
            actions = torch.from_numpy(actions).float()
            
            # Normalize actions if needed
            if self.norm_action:
                actions = self._normalize_actions(actions)
            
            # Prepare output
            output = {
                'images': images[:self.window_size],  # History images
                'actions': actions,  # All actions (history + future)
                'instruction': episode['instruction'],
                'scenario': episode['scenario'],
            }
            
            return output
    
    def _normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Normalize actions to [-1, 1] range (Mobile VLA 2D standard)"""
        normalized = torch.zeros_like(actions)
        
        # Normalize each dimension (Mobile VLA 2D bounds)
        bounds = [
            self.action_bounds['linear_x'],
            self.action_bounds['linear_y'],
        ]
        
        for i, (min_val, max_val) in enumerate(bounds):
            normalized[:, i] = 2 * (actions[:, i] - min_val) / (max_val - min_val) - 1
            normalized[:, i] = torch.clamp(normalized[:, i], -1, 1)
        
        return normalized
    
    def _denormalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Denormalize actions from [-1, 1] to original range (Mobile VLA 2D standard)"""
        denormalized = torch.zeros_like(actions)
        
        bounds = [
            self.action_bounds['linear_x'],
            self.action_bounds['linear_y'],
        ]
        
        for i, (min_val, max_val) in enumerate(bounds):
            denormalized[:, i] = (actions[:, i] + 1) / 2 * (max_val - min_val) + min_val
        
        return denormalized
    
    def collater(self, batch: List[Dict]) -> Dict[str, Any]:
        """
        Collate batch of samples
        
        Args:
            batch: List of samples from __getitem__
            
        Returns:
            Batched dict compatible with RoboVLMs trainer
        """
        # Stack images
        images = torch.stack([item['images'] for item in batch])  # (B, T, H, W, 3)
        
        # Stack actions
        actions = torch.stack([item['actions'] for item in batch])  # (B, T, 4)
        
        # Collect instructions
        instructions = [item['instruction'] for item in batch]
        scenarios = [item['scenario'] for item in batch]
        
        return {
            'images': images,
            'actions': actions,
            'text': instructions,
            'scenarios': scenarios,
        }


def get_mobile_vla_dataset(config: Dict) -> Tuple[MobileVLADataset, MobileVLADataset]:
    """
    Create train and validation datasets
    
    Args:
        config: Configuration dict with dataset parameters
        
    Returns:
        (train_dataset, val_dataset)
    """
    train_config = config.get('train_dataset', {})
    val_config = config.get('val_dataset', {})
    
    # Create datasets
    train_dataset = MobileVLADataset(
        mode='train',
        split='train',
        **train_config
    )
    
    val_dataset = MobileVLADataset(
        mode='val',
        split='val',
        **val_config
    )
    
    return train_dataset, val_dataset

