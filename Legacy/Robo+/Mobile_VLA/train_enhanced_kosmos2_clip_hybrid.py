#!/usr/bin/env python3
"""
Enhanced Kosmos2+CLIP Hybrid Model Training Script
Vision Resampler를 통합한 향상된 하이브리드 모델 학습

주요 기능:
1. Vision Resampler 통합 모델 학습
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

from enhanced_kosmos2_clip_hybrid import EnhancedKosmos2CLIPHybrid, EnhancedKosmos2CLIPHybridTrainer

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Original72EpisodesDataset(Dataset):
    """
    Original 72 episodes HDF5 dataset for enhanced model training
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
        
        # Find HDF5 files
        self.hdf5_files = list(self.data_dir.glob("*.h5"))
        logger.info(f"Found {len(self.hdf5_files)} HDF5 files")
        
        # Load all episodes
        self.episodes = []
        self._load_episodes()
        
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
        T = len(episode['images'])
        frame_idx = np.random.randint(0, T)
        
        # Get image
        image = episode['images'][frame_idx]  # [H, W, C]
        image = Image.fromarray(image.astype(np.uint8))
        
        if self.transform:
            image = self.transform(image)
        
        # Get action
        action = episode['actions'][frame_idx]  # [action_dim]
        action = torch.tensor(action, dtype=torch.float32)
        
        # Get text
        text = episode['texts'][frame_idx]
        if isinstance(text, bytes):
            text = text.decode('utf-8')
        text = str(text)
        
        return {
            'image': image,
            'action': action,
            'text': text,
            'episode_idx': idx,
            'frame_idx': frame_idx
        }


def create_data_transforms():
    """Create data transforms for training"""
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # Note: No normalization here as Kosmos2 processor handles it
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    return train_transform, val_transform


def train_enhanced_model(
    data_dir: str = "/home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset",
    output_dir: str = "enhanced_kosmos2_clip_hybrid_results",
    epochs: int = 10,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Train Enhanced Kosmos2+CLIP Hybrid Model
    
    Args:
        data_dir: Path to HDF5 dataset
        output_dir: Output directory for results
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device to use for training
    """
    logger.info("Starting Enhanced Kosmos2+CLIP Hybrid Model training...")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Create data transforms
    train_transform, val_transform = create_data_transforms()
    
    # Create datasets
    logger.info("Creating datasets...")
    train_dataset = Original72EpisodesDataset(
        data_dir=data_dir,
        split="train",
        transform=train_transform
    )
    
    # Split dataset (80% train, 20% val)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    # Update val dataset transform
    val_dataset.dataset.transform = val_transform
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing issues
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=custom_collate_fn
    )
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Val dataset size: {len(val_dataset)}")
    
    # Create model
    logger.info("Creating enhanced model...")
    model = EnhancedKosmos2CLIPHybrid(
        action_dim=2,  # 2D 액션 (linear_x, linear_y) - Z값은 항상 0
        vision_resampler_tokens=64,
        hidden_dim=768,
        lstm_hidden_dim=256,
        lstm_num_layers=2,
        dropout=0.1,
        mobile_optimized=True
    )
    
    # Get model info
    model_info = model.get_model_info()
    logger.info(f"Model info: {model_info}")
    
    # Create trainer
    trainer = EnhancedKosmos2CLIPHybridTrainer(
        model=model,
        learning_rate=learning_rate,
        weight_decay=1e-5,
        device=device
    )
    
    # Training loop
    logger.info("Starting training...")
    best_val_mae = float('inf')
    training_history = []
    
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch+1}/{epochs}")
        
        # Training phase
        model.train()
        train_metrics = {'loss': 0.0, 'mae': 0.0, 'lr': 0.0}
        train_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            try:
                # Get batch data
                images = batch['image']
                texts = batch['text']
                actions = batch['action']
                
                # Training step
                metrics = trainer.train_step(images, texts, actions)
                
                # Accumulate metrics
                for key in train_metrics:
                    train_metrics[key] += metrics[key]
                train_batches += 1
                
                # Log progress
                if batch_idx % 10 == 0:
                    logger.info(f"  Batch {batch_idx}/{len(train_loader)}: "
                              f"Loss={metrics['loss']:.4f}, MAE={metrics['mae']:.4f}")
                
            except Exception as e:
                logger.warning(f"Training batch {batch_idx} failed: {e}")
                continue
        
        # Average training metrics
        for key in train_metrics:
            train_metrics[key] /= train_batches
        
        # Validation phase
        model.eval()
        val_metrics = {'val_loss': 0.0, 'val_mae': 0.0}
        val_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                try:
                    # Get batch data
                    images = batch['image']
                    texts = batch['text']
                    actions = batch['action']
                    
                    # Validation step
                    metrics = trainer.validate(images, texts, actions)
                    
                    # Accumulate metrics
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
        trainer.scheduler.step()
        
        # Log epoch results
        logger.info(f"Epoch {epoch+1} Results:")
        logger.info(f"  Train Loss: {train_metrics['loss']:.4f}, Train MAE: {train_metrics['mae']:.4f}")
        logger.info(f"  Val Loss: {val_metrics['val_loss']:.4f}, Val MAE: {val_metrics['val_mae']:.4f}")
        logger.info(f"  Learning Rate: {train_metrics['lr']:.6f}")
        
        # Save epoch history
        epoch_history = {
            'epoch': epoch + 1,
            'train_loss': train_metrics['loss'],
            'train_mae': train_metrics['mae'],
            'val_loss': val_metrics['val_loss'],
            'val_mae': val_metrics['val_mae'],
            'lr': train_metrics['lr']
        }
        training_history.append(epoch_history)
        
        # Save best model
        if val_metrics['val_mae'] < best_val_mae:
            best_val_mae = val_metrics['val_mae']
            best_model_path = output_path / "best_enhanced_kosmos2_clip_hybrid.pth"
            trainer.save_checkpoint(
                str(best_model_path),
                epoch + 1,
                {**train_metrics, **val_metrics}
            )
            logger.info(f"New best model saved with Val MAE: {best_val_mae:.4f}")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = output_path / f"enhanced_kosmos2_clip_hybrid_epoch_{epoch+1}.pth"
            trainer.save_checkpoint(
                str(checkpoint_path),
                epoch + 1,
                {**train_metrics, **val_metrics}
            )
    
    # Save final model
    final_model_path = output_path / "final_enhanced_kosmos2_clip_hybrid.pth"
    trainer.save_checkpoint(
        str(final_model_path),
        epochs,
        {**train_metrics, **val_metrics}
    )
    
    # Save training history
    history_path = output_path / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    # Save model info
    info_path = output_path / "model_info.json"
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    logger.info(f"Training completed!")
    logger.info(f"Best validation MAE: {best_val_mae:.4f}")
    logger.info(f"Results saved to: {output_path}")
    
    return best_val_mae, training_history


def custom_collate_fn(batch):
    """Custom collate function for DataLoader"""
    images = torch.stack([item['image'] for item in batch])
    actions = torch.stack([item['action'] for item in batch])
    texts = [item['text'] for item in batch]
    
    return {
        'image': images,
        'action': actions,
        'text': texts
    }


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Train Enhanced Kosmos2+CLIP Hybrid Model")
    parser.add_argument("--data_dir", type=str, default="/home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset",
                       help="Path to HDF5 dataset")
    parser.add_argument("--output_dir", type=str, default="enhanced_kosmos2_clip_hybrid_results",
                       help="Output directory for results")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use for training")
    
    args = parser.parse_args()
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        logger.error(f"Data directory {args.data_dir} does not exist!")
        return
    
    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        args.device = "cpu"
    
    logger.info(f"Training configuration:")
    logger.info(f"  Data directory: {args.data_dir}")
    logger.info(f"  Output directory: {args.output_dir}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(f"  Device: {args.device}")
    
    # Start training
    try:
        best_mae, history = train_enhanced_model(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=args.device
        )
        
        logger.info(f"Training completed successfully!")
        logger.info(f"Best validation MAE: {best_mae:.4f}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
