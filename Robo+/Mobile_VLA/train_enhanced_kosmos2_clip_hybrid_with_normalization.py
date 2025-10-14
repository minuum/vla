#!/usr/bin/env python3
"""
Enhanced Kosmos2+CLIP Hybrid Model with CLIP Normalization Training Script
Vision Resampler와 CLIP Normalization을 통합한 향상된 하이브리드 모델 학습

주요 기능:
1. Vision Resampler + CLIP Normalization 통합 모델 학습
2. 원본 72 에피소드 데이터셋 사용
3. 성능 향상 검증
4. 모바일 최적화
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

from enhanced_kosmos2_clip_hybrid_with_normalization import (
    EnhancedKosmos2CLIPHybridWithNormalization, 
    EnhancedKosmos2CLIPHybridWithNormalizationTrainer
)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Original72EpisodesDataset(Dataset):
    """
    Original 72 episodes HDF5 dataset for enhanced model with normalization training
    """
    
    def __init__(
        self, 
        data_dir: str = "/home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset",
        split: str = "train",
        transform=None
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        
        if not self.data_dir.exists():
            logger.error(f"Data directory {self.data_dir} does not exist!")
            raise FileNotFoundError(f"Data directory {self.data_dir} does not exist!")
        
        # Load HDF5 files
        self.hdf5_files = list(self.data_dir.glob("*.h5"))
        logger.info(f"Found {len(self.hdf5_files)} HDF5 files")
        
        # Load episodes
        self.episodes = []
        self._load_episodes()
        
        # Split data
        if split == "train":
            self.episodes = self.episodes[:int(len(self.episodes) * 0.8)]
        else:
            self.episodes = self.episodes[int(len(self.episodes) * 0.8):]
        
        logger.info(f"Loaded {len(self.episodes)} episodes for {split} split")
    
    def _load_episodes(self):
        """Load all episodes from HDF5 files"""
        for hdf5_file in self.hdf5_files:
            try:
                with h5py.File(hdf5_file, 'r') as f:
                    # Get episode data
                    images = f['images'][:]  # [T, H, W, C]
                    actions = f['actions'][:]  # [T, action_dim]
                    
                    # Handle different text key names
                    if 'texts' in f:
                        texts = f['texts'][:]  # [T] or [T, max_text_len]
                    elif 'action_event_types' in f:
                        texts = f['action_event_types'][:]  # [T] or [T, max_text_len]
                    else:
                        # Generate default text commands
                        texts = np.array(['go forward'] * len(images))
                    
                    # Convert texts to strings if needed
                    if texts.dtype != object:
                        texts = [text.decode('utf-8') if isinstance(text, bytes) else str(text) for text in texts]
                    
                    # Create episode
                    episode = {
                        'images': images,
                        'actions': actions,
                        'texts': texts,
                        'file': str(hdf5_file)
                    }
                    
                    self.episodes.append(episode)
                    
            except Exception as e:
                logger.warning(f"Failed to load {hdf5_file}: {e}")
                continue
    
    def __len__(self):
        return len(self.episodes)
    
    def __getitem__(self, idx):
        episode = self.episodes[idx]
        
        # Get random frame from episode
        frame_idx = np.random.randint(0, len(episode['images']))
        
        # Get image
        image = episode['images'][frame_idx]  # [H, W, C]
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        # Get action
        action = episode['actions'][frame_idx]  # [action_dim]
        action = torch.tensor(action, dtype=torch.float32)
        
        # Get text
        text = episode['texts'][frame_idx]
        if isinstance(text, (list, np.ndarray)):
            text = str(text[0]) if len(text) > 0 else "go forward"
        else:
            text = str(text)
        
        return {
            'image': image,
            'action': action,
            'text': text,
            'episode_file': episode['file']
        }

def get_transforms():
    """Get image transforms"""
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    return train_transform, val_transform


def train_enhanced_model_with_normalization(
    data_dir: str = "/home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset",
    output_dir: str = "enhanced_kosmos2_clip_hybrid_with_normalization_results",
    epochs: int = 10,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    normalization_type: str = "mobile"  # "mobile" or "adaptive"
):
    """Train enhanced model with normalization"""
    
    logger.info("Starting Enhanced Kosmos2+CLIP Hybrid Model with Normalization training...")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Create datasets
    logger.info("Creating datasets...")
    train_transform, val_transform = get_transforms()
    
    train_dataset = Original72EpisodesDataset(
        data_dir=data_dir,
        split="train",
        transform=train_transform
    )
    
    val_dataset = Original72EpisodesDataset(
        data_dir=data_dir,
        split="val",
        transform=val_transform
    )
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Val dataset size: {len(val_dataset)}")
    
    # Create model
    logger.info("Creating enhanced model with normalization...")
    model = EnhancedKosmos2CLIPHybridWithNormalization(
        action_dim=2,  # 2D 액션 (linear_x, linear_y) - Z값은 항상 0
        vision_resampler_tokens=64,
        hidden_dim=768,
        lstm_hidden_dim=256,
        lstm_num_layers=2,
        dropout=0.1,
        mobile_optimized=True,
        use_clip_normalization=True,
        normalization_type=normalization_type
    )
    
    # Get model info
    model_info = model.get_model_info()
    logger.info(f"Model info: {model_info}")
    
    # Create trainer
    trainer = EnhancedKosmos2CLIPHybridWithNormalizationTrainer(model, device)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        collate_fn=lambda batch: {
            'images': torch.stack([item['image'] for item in batch]),
            'actions': torch.stack([item['action'] for item in batch]),
            'texts': [item['text'] for item in batch]
        }
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: {
            'images': torch.stack([item['image'] for item in batch]),
            'actions': torch.stack([item['action'] for item in batch]),
            'texts': [item['text'] for item in batch]
        }
    )
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training loop
    logger.info("Starting training...")
    best_val_mae = float('inf')
    history = {
        'train_loss': [],
        'train_mae': [],
        'val_loss': [],
        'val_mae': [],
        'learning_rate': []
    }
    
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch+1}/{epochs}")
        
        # Training
        model.train()
        train_metrics = {'loss': 0.0, 'mae': 0.0}
        train_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            try:
                images = batch['images']
                texts = batch['texts']
                actions = batch['actions']
                
                # Move to device
                images = images.to(device)
                actions = actions.to(device)
                
                # Training step
                metrics = trainer.train_step(images, texts, actions)
                
                # Backward pass
                optimizer.zero_grad()
                loss = trainer.criterion(
                    model(images, texts), 
                    actions
                )
                loss.backward()
                optimizer.step()
                
                # Update metrics
                for key in train_metrics:
                    train_metrics[key] += metrics[key]
                train_batches += 1
                
                if batch_idx % 10 == 0:
                    logger.info(f"  Batch {batch_idx}/{len(train_loader)}: Loss={metrics['loss']:.4f}, MAE={metrics['mae']:.4f}")
                    
            except Exception as e:
                logger.warning(f"Training batch {batch_idx} failed: {e}")
                continue
        
        # Average training metrics
        for key in train_metrics:
            train_metrics[key] /= train_batches
        
        # Validation
        model.eval()
        val_metrics = {'loss': 0.0, 'mae': 0.0}
        val_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                try:
                    images = batch['images']
                    texts = batch['texts']
                    actions = batch['actions']
                    
                    # Move to device
                    images = images.to(device)
                    actions = actions.to(device)
                    
                    # Validation step
                    metrics = trainer.validate_step(images, texts, actions)
                    
                    # Update metrics
                    for key in val_metrics:
                        val_metrics[key] += metrics[key]
                    val_batches += 1
                    
                except Exception as e:
                    logger.warning(f"Validation batch {batch_idx} failed: {e}")
                    continue
        
        # Average validation metrics
        for key in val_metrics:
            val_metrics[key] /= val_batches
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Log epoch results
        logger.info(f"Epoch {epoch+1} Results:")
        logger.info(f"  Train Loss: {train_metrics['loss']:.4f}, Train MAE: {train_metrics['mae']:.4f}")
        logger.info(f"  Val Loss: {val_metrics['loss']:.4f}, Val MAE: {val_metrics['mae']:.4f}")
        logger.info(f"  Learning Rate: {current_lr:.6f}")
        
        # Save history
        history['train_loss'].append(train_metrics['loss'])
        history['train_mae'].append(train_metrics['mae'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_mae'].append(val_metrics['mae'])
        history['learning_rate'].append(current_lr)
        
        # Save best model
        if val_metrics['mae'] < best_val_mae:
            best_val_mae = val_metrics['mae']
            best_model_path = output_path / f"best_enhanced_kosmos2_clip_hybrid_with_{normalization_type}_normalization.pth"
            trainer.save_checkpoint(str(best_model_path), epoch, val_metrics)
            logger.info(f"New best model saved with Val MAE: {best_val_mae:.4f}")
        
        # Save epoch checkpoint
        if (epoch + 1) % 5 == 0:
            epoch_path = output_path / f"enhanced_kosmos2_clip_hybrid_with_{normalization_type}_normalization_epoch_{epoch+1}.pth"
            trainer.save_checkpoint(str(epoch_path), epoch, val_metrics)
    
    # Save final model
    final_path = output_path / f"final_enhanced_kosmos2_clip_hybrid_with_{normalization_type}_normalization.pth"
    trainer.save_checkpoint(str(final_path), epochs-1, val_metrics)
    
    # Save training history
    history_path = output_path / f"training_history_{normalization_type}_normalization.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info("Training completed!")
    logger.info(f"Best validation MAE: {best_val_mae:.4f}")
    logger.info(f"Results saved to: {output_dir}")
    
    return best_val_mae, history


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Train Enhanced Kosmos2+CLIP Hybrid Model with Normalization")
    parser.add_argument("--data_dir", type=str, default="/home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset",
                       help="Path to HDF5 dataset")
    parser.add_argument("--output_dir", type=str, default="enhanced_kosmos2_clip_hybrid_with_normalization_results",
                       help="Output directory for results")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--normalization_type", type=str, default="mobile", choices=["mobile", "adaptive"],
                       help="Type of CLIP normalization to use")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use for training")
    
    args = parser.parse_args()
    
    logger.info("Training configuration:")
    logger.info(f"  Data directory: {args.data_dir}")
    logger.info(f"  Output directory: {args.output_dir}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(f"  Normalization type: {args.normalization_type}")
    logger.info(f"  Device: {args.device}")
    
    try:
        best_mae, history = train_enhanced_model_with_normalization(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=args.device,
            normalization_type=args.normalization_type
        )
        
        logger.info("Training completed successfully!")
        logger.info(f"Best validation MAE: {best_mae:.4f}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
