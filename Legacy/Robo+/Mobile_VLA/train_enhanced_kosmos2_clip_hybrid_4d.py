#!/usr/bin/env python3
"""
Enhanced Kosmos2+CLIP Hybrid Model Training Script (4D Actions)
RoboVLMs 스타일 4D 액션 공간으로 확장된 모델 학습

주요 기능:
1. 4D 액션 공간 (linear_x, linear_y, angular_z, action_type)
2. 액션 타입 분류 헤드
3. 다중 손실 함수
4. RoboVLMs 스타일 학습
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import h5py
from PIL import Image
import torchvision.transforms as transforms

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_kosmos2_clip_hybrid import EnhancedKosmos2CLIPHybrid, ACTION_TYPES

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RoboVLMsMobileDataset(Dataset):
    """
    RoboVLMs 스타일 Mobile VLA 데이터셋
    4D 액션 공간과 액션 타입을 지원
    """
    
    def __init__(
        self, 
        data_dir: str = "/home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset",
        split: str = "train",
        transform=None
    ):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        # Load episodes
        self.episodes = self._load_episodes()
        
        logger.info(f"RoboVLMs Mobile Dataset loaded:")
        logger.info(f"  - Split: {split}")
        logger.info(f"  - Episodes: {len(self.episodes)}")
        logger.info(f"  - Action types: {len(ACTION_TYPES)}")
    
    def _load_episodes(self) -> List[Dict]:
        """Load episodes from HDF5 files"""
        episodes = []
        
        if not os.path.exists(self.data_dir):
            logger.error(f"Data directory {self.data_dir} does not exist!")
            return episodes
        
        h5_files = [f for f in os.listdir(self.data_dir) if f.endswith('.h5')]
        
        for h5_file in h5_files:
            file_path = os.path.join(self.data_dir, h5_file)
            try:
                with h5py.File(file_path, 'r') as f:
                    # Extract scenario from filename
                    scenario = self._extract_scenario(h5_file)
                    
                    episode = {
                        'file_path': file_path,
                        'scenario': scenario,
                        'images': f['images'][:],
                        'actions': f['actions'][:],
                        'action_event_types': f['action_event_types'][:]
                    }
                    episodes.append(episode)
                    
            except Exception as e:
                logger.warning(f"Failed to load {h5_file}: {e}")
                continue
        
        return episodes
    
    def _extract_scenario(self, filename: str) -> str:
        """Extract scenario from filename"""
        # episode_20250815_122923_2box_vert_left_core_medium.h5
        parts = filename.split('_')
        if len(parts) >= 7:
            return f"{parts[3]}_{parts[4]}_{parts[5]}_{parts[7].split('.')[0]}"
        return "unknown"
    
    def _convert_3d_to_4d_actions(self, actions_3d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        3D 액션을 4D 액션과 액션 타입으로 변환
        
        Args:
            actions_3d: [T, 3] (linear_x, linear_y, angular_z)
            
        Returns:
            actions_4d: [T, 4] (linear_x, linear_y, angular_z, action_type)
            action_types: [T] 액션 타입 인덱스
        """
        batch_size = actions_3d.shape[0]
        actions_4d = np.zeros((batch_size, 4))
        action_types = np.zeros(batch_size, dtype=np.int64)
        
        for i, action in enumerate(actions_3d):
            linear_x, linear_y, angular_z = action
            
            # 액션 타입 결정
            if abs(linear_x) > 0.1:
                if linear_x > 0:
                    action_type = 0  # move_forward
                else:
                    action_type = 1  # move_backward
            elif abs(linear_y) > 0.1:
                if linear_y > 0:
                    action_type = 5  # move_right
                else:
                    action_type = 4  # move_left
            elif abs(angular_z) > 0.1:
                if angular_z > 0:
                    action_type = 3  # turn_right
                else:
                    action_type = 2  # turn_left
            else:
                action_type = 6  # stop
            
            # 4D 액션 구성
            actions_4d[i] = [linear_x, linear_y, angular_z, action_type]
            action_types[i] = action_type
        
        return actions_4d, action_types
    
    def __len__(self):
        return len(self.episodes)
    
    def __getitem__(self, idx):
        episode = self.episodes[idx]
        
        # Load data
        with h5py.File(episode['file_path'], 'r') as f:
            images = f['images'][:]
            actions_3d = f['actions'][:]
        
        # Convert 3D to 4D actions
        actions_4d, action_types = self._convert_3d_to_4d_actions(actions_3d)
        
        # Select random frame
        frame_idx = np.random.randint(0, len(images))
        
        # Process image
        image = images[frame_idx]
        
        # 이미지 전처리 (uint8 [0, 255] → PIL → Tensor → 정규화)
        if self.transform:
            image = self.transform(image)
        
        # Process actions
        action_4d = actions_4d[frame_idx]
        action_type = action_types[frame_idx]
        
        # Generate text instruction
        text = self._generate_text_instruction(episode['scenario'])
        
        return {
            'images': image,
            'actions': torch.tensor(action_4d, dtype=torch.float32),
            'action_types': torch.tensor(action_type, dtype=torch.long),
            'texts': text,
            'scenario': episode['scenario']
        }
    
    def _generate_text_instruction(self, scenario: str) -> str:
        """Generate text instruction from scenario"""
        # 기본 명령 (can tracking/navigation)
        base_instruction = "start_action"
        
        # 시나리오별 추가 설명
        if "1box" in scenario:
            base_instruction += " - single box scenario"
        elif "2box" in scenario:
            base_instruction += " - dual box scenario"
        
        if "vert" in scenario:
            base_instruction += " - vertical orientation"
        elif "hori" in scenario:
            base_instruction += " - horizontal orientation"
        
        if "left" in scenario:
            base_instruction += " - left position"
        elif "right" in scenario:
            base_instruction += " - right position"
        
        if "close" in scenario:
            base_instruction += " - close distance"
        elif "medium" in scenario:
            base_instruction += " - medium distance"
        elif "far" in scenario:
            base_instruction += " - far distance"
        
        return base_instruction

class RoboVLMsMobileLoss(nn.Module):
    """
    RoboVLMs 스타일 다중 손실 함수
    """
    
    def __init__(self, action_weight: float = 1.0, type_weight: float = 0.3):
        super().__init__()
        self.action_weight = action_weight
        self.type_weight = type_weight
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, predictions, targets):
        # 액션 회귀 손실
        action_loss = self.mse_loss(predictions["actions"], targets["actions"])
        
        # 액션 타입 분류 손실
        type_loss = self.ce_loss(predictions["action_types"], targets["action_types"])
        
        # 총 손실
        total_loss = self.action_weight * action_loss + self.type_weight * type_loss
        
        return {
            "total_loss": total_loss,
            "action_loss": action_loss,
            "type_loss": type_loss
        }

def train_robovlms_mobile_model(
    model: EnhancedKosmos2CLIPHybrid,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 5,
    learning_rate: float = 1e-4,
    device: str = "cuda"
):
    """Train RoboVLMs Mobile model"""
    
    model = model.to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4
    )
    
    # Loss function
    criterion = RoboVLMsMobileLoss()
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=epochs,
        eta_min=1e-6
    )
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            try:
                images = batch['images'].to(device)
                actions = batch['actions'].to(device)
                action_types = batch['action_types'].to(device)
                texts = batch['texts']
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(images, texts)
                predicted_actions = outputs["actions"]
                predicted_types = outputs["action_types"]
                
                # Calculate loss
                loss_dict = criterion(
                    {"actions": predicted_actions, "action_types": predicted_types},
                    {"actions": actions, "action_types": action_types}
                )
                
                loss = loss_dict["total_loss"]
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_batches += 1
                
                if batch_idx % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, "
                              f"Loss: {loss.item():.4f}, "
                              f"Action Loss: {loss_dict['action_loss'].item():.4f}, "
                              f"Type Loss: {loss_dict['type_loss'].item():.4f}")
                
            except Exception as e:
                logger.error(f"Training batch {batch_idx} failed: {e}")
                continue
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                try:
                    images = batch['images'].to(device)
                    actions = batch['actions'].to(device)
                    action_types = batch['action_types'].to(device)
                    texts = batch['texts']
                    
                    # Forward pass
                    outputs = model(images, texts)
                    predicted_actions = outputs["actions"]
                    predicted_types = outputs["action_types"]
                    
                    # Calculate loss
                    loss_dict = criterion(
                        {"actions": predicted_actions, "action_types": predicted_types},
                        {"actions": actions, "action_types": action_types}
                    )
                    
                    val_loss += loss_dict["total_loss"].item()
                    val_batches += 1
                    
                except Exception as e:
                    logger.error(f"Validation batch {batch_idx} failed: {e}")
                    continue
        
        # Calculate average losses
        avg_train_loss = train_loss / max(train_batches, 1)
        avg_val_loss = val_loss / max(val_batches, 1)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Update learning rate
        scheduler.step()
        
        logger.info(f"Epoch {epoch+1}/{epochs} completed:")
        logger.info(f"  Train Loss: {avg_train_loss:.4f}")
        logger.info(f"  Val Loss: {avg_val_loss:.4f}")
        logger.info(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': avg_val_loss,
                'train_loss': avg_train_loss,
                'action_types': ACTION_TYPES
            }, f'best_robovlms_mobile_model_epoch_{epoch+1}.pt')
            logger.info(f"Best model saved with val loss: {avg_val_loss:.4f}")
    
    return train_losses, val_losses

def main():
    parser = argparse.ArgumentParser(description='Train RoboVLMs Mobile Model')
    parser.add_argument('--data_dir', type=str, 
                       default='/home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset',
                       help='Data directory')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    
    args = parser.parse_args()
    
    # Device setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),  # uint8 [0, 255] → PIL Image
        transforms.Resize((224, 224)),
        transforms.ToTensor(),    # PIL Image → float [0, 1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = RoboVLMsMobileDataset(
        data_dir=args.data_dir,
        split='train',
        transform=transform
    )
    
    val_dataset = RoboVLMsMobileDataset(
        data_dir=args.data_dir,
        split='val',
        transform=transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )
    
    # Create model
    model = EnhancedKosmos2CLIPHybrid(
        action_dim=4,  # 4D 액션 공간
        vision_resampler_tokens=64,
        hidden_dim=768,
        lstm_hidden_dim=256,
        dropout=0.1,
        mobile_optimized=True
    )
    
    # Print model info
    model_info = model.get_model_info()
    logger.info(f"Model Info: {model_info}")
    
    # Train model
    train_losses, val_losses = train_robovlms_mobile_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=device
    )
    
    logger.info("Training completed!")
    logger.info(f"Final train loss: {train_losses[-1]:.4f}")
    logger.info(f"Final val loss: {val_losses[-1]:.4f}")

if __name__ == "__main__":
    main()
