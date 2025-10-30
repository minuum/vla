#!/usr/bin/env python3
"""
Training Script for Enhanced Kosmos2+CLIP Hybrid Model without Claw Matrix
Vision Resampler, CLIP Normalization만 사용한 안정적인 모델 학습

주요 기능:
1. 안정적인 학습
2. 성능 모니터링
3. 체크포인트 저장
4. 차원 문제 해결
"""

import os
import sys
import logging
import argparse
import time
from typing import Dict, List, Tuple, Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import h5py
from tqdm import tqdm
import json

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_kosmos2_clip_hybrid_with_normalization import EnhancedKosmos2CLIPHybridWithNormalization

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Original72EpisodesDataset(Dataset):
    """Original 72 episodes dataset"""
    
    def __init__(self, data_dir: str = "/home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset"):
        self.data_dir = data_dir
        self.episodes = self._load_episodes()
        logger.info(f"Loaded {len(self.episodes)} episodes from {data_dir}")
    
    def _load_episodes(self) -> List[Dict]:
        """Load all episodes from the dataset"""
        episodes = []
        
        if not os.path.exists(self.data_dir):
            logger.error(f"Data directory {self.data_dir} does not exist!")
            return episodes
        
        # Load all .h5 files
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.h5'):
                filepath = os.path.join(self.data_dir, filename)
                try:
                    with h5py.File(filepath, 'r') as f:
                        # Check available keys
                        available_keys = list(f.keys())
                        logger.info(f"Available keys in {filename}: {available_keys}")
                        
                        # Load data
                        images = f['images'][:]  # [T, H, W, C]
                        actions = f['actions'][:]  # [T, action_dim]
                        
                        # Handle text commands
                        if 'texts' in f:
                            texts = f['texts'][:]
                        elif 'action_event_types' in f:
                            texts = f['action_event_types'][:]
                        else:
                            # Default text command
                            texts = np.array(['go forward'] * len(images))
                        
                        # Convert to list of strings
                        if isinstance(texts[0], bytes):
                            texts = [t.decode('utf-8') for t in texts]
                        else:
                            texts = [str(t) for t in texts]
                        
                        episodes.append({
                            'images': images,
                            'actions': actions,
                            'texts': texts,
                            'filename': filename
                        })
                        
                except Exception as e:
                    logger.warning(f"Failed to load {filename}: {e}")
                    continue
        
        return episodes
    
    def __len__(self):
        return len(self.episodes)
    
    def __getitem__(self, idx):
        episode = self.episodes[idx]
        
        # Get random frame from episode
        frame_idx = np.random.randint(0, len(episode['images']))
        
        # Get image and action
        image = episode['images'][frame_idx]  # [H, W, C]
        action = episode['actions'][frame_idx]  # [action_dim]
        text = episode['texts'][frame_idx]  # string
        
        # Convert image to tensor and normalize
        image = torch.from_numpy(image).float() / 255.0  # [H, W, C]
        image = image.permute(2, 0, 1)  # [C, H, W]
        
        # Convert action to tensor
        action = torch.from_numpy(action).float()
        
        return {
            'images': image,
            'actions': action,
            'texts': text
        }

def train_enhanced_model_without_claw_matrix(
    model: EnhancedKosmos2CLIPHybridWithNormalization,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    epochs: int = 5,
    learning_rate: float = 1e-4,
    save_dir: str = "checkpoints"
):
    """Train the enhanced model without claw matrix"""
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training history
    history = {
        'train_loss': [],
        'train_mae': [],
        'val_loss': [],
        'val_mae': [],
        'learning_rates': []
    }
    
    best_val_mae = float('inf')
    best_epoch = 0
    
    logger.info(f"Starting training for {epochs} epochs...")
    logger.info(f"Device: {device}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Save directory: {save_dir}")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_losses = []
        train_maes = []
        
        logger.info(f"Epoch {epoch+1}/{epochs} - Training...")
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            try:
                # Move data to device
                images = batch['images'].to(device)
                actions = batch['actions'].to(device)
                texts = batch['texts']
                
                # Forward pass
                optimizer.zero_grad()
                predicted_actions = model(images, texts)
                
                # Compute loss
                loss = F.mse_loss(predicted_actions, actions)
                mae = F.l1_loss(predicted_actions, actions).item()
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_losses.append(loss.item())
                train_maes.append(mae)
                
                # Log progress
                if batch_idx % 10 == 0:
                    logger.info(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}, MAE: {mae:.4f}")
                
            except Exception as e:
                logger.warning(f"Training batch {batch_idx} failed: {e}")
                continue
        
        # Validation phase
        model.eval()
        val_losses = []
        val_maes = []
        
        logger.info(f"Epoch {epoch+1}/{epochs} - Validation...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validation")):
                try:
                    # Move data to device
                    images = batch['images'].to(device)
                    actions = batch['actions'].to(device)
                    texts = batch['texts']
                    
                    # Forward pass
                    predicted_actions = model(images, texts)
                    
                    # Compute loss
                    loss = F.mse_loss(predicted_actions, actions)
                    mae = F.l1_loss(predicted_actions, actions).item()
                    
                    val_losses.append(loss.item())
                    val_maes.append(mae)
                    
                except Exception as e:
                    logger.warning(f"Validation batch {batch_idx} failed: {e}")
                    continue
        
        # Calculate epoch metrics
        avg_train_loss = np.mean(train_losses) if train_losses else 0.0
        avg_train_mae = np.mean(train_maes) if train_maes else 0.0
        avg_val_loss = np.mean(val_losses) if val_losses else 0.0
        avg_val_mae = np.mean(val_maes) if val_maes else 0.0
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['train_mae'].append(avg_train_mae)
        history['val_loss'].append(avg_val_loss)
        history['val_mae'].append(avg_val_mae)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Log epoch results
        logger.info(f"Epoch {epoch+1}/{epochs} Results:")
        logger.info(f"  Train Loss: {avg_train_loss:.4f}, Train MAE: {avg_train_mae:.4f}")
        logger.info(f"  Val Loss: {avg_val_loss:.4f}, Val MAE: {avg_val_mae:.4f}")
        logger.info(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if avg_val_mae < best_val_mae:
            best_val_mae = avg_val_mae
            best_epoch = epoch + 1
            
            # Save checkpoint
            checkpoint_path = os.path.join(save_dir, f"best_model_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'train_mae': avg_train_mae,
                'val_loss': avg_val_loss,
                'val_mae': avg_val_mae,
                'model_info': model.get_model_info(),
                'history': history
            }, checkpoint_path)
            
            logger.info(f"New best model saved: {checkpoint_path}")
            logger.info(f"Best validation MAE: {best_val_mae:.4f}")
        
        # Update scheduler
        scheduler.step()
        
        # Save history
        history_path = os.path.join(save_dir, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
    
    logger.info(f"Training completed!")
    logger.info(f"Best validation MAE: {best_val_mae:.4f} at epoch {best_epoch}")
    
    return history, best_val_mae

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train Enhanced Kosmos2+CLIP Hybrid Model without Claw Matrix")
    parser.add_argument("--data_dir", type=str, default="/home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset",
                        help="Path to the dataset directory")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--save_dir", type=str, default="checkpoints_enhanced", help="Directory to save checkpoints")
    parser.add_argument("--normalization_type", type=str, default="mobile", choices=["mobile", "adaptive"],
                        help="CLIP normalization type")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (auto, cuda, cpu)")
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")
    
    # Check CUDA availability
    if device == "cuda":
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA device: {torch.cuda.get_device_name()}")
            logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create model
    logger.info("Creating Enhanced Kosmos2+CLIP Hybrid Model with Normalization...")
    model = EnhancedKosmos2CLIPHybridWithNormalization(
        action_dim=2,  # 2D 액션 (linear_x, linear_y) - Z값은 항상 0
        vision_resampler_tokens=64,
        mobile_optimized=True,
        use_clip_normalization=True,
        normalization_type=args.normalization_type
    )
    
    # Get model info
    model_info = model.get_model_info()
    logger.info(f"Model info: {model_info}")
    
    # Move model to device
    model = model.to(device)
    
    # Create dataset
    logger.info("Loading dataset...")
    dataset = Original72EpisodesDataset(data_dir=args.data_dir)
    
    if len(dataset) == 0:
        logger.error("No episodes loaded! Check data directory.")
        return
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    logger.info(f"Dataset split: {len(train_dataset)} train, {len(val_dataset)} validation")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True if device == "cuda" else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if device == "cuda" else False
    )
    
    # Start training
    logger.info("Starting training...")
    start_time = time.time()
    
    history, best_val_mae = train_enhanced_model_without_claw_matrix(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        save_dir=args.save_dir
    )
    
    end_time = time.time()
    training_time = end_time - start_time
    
    logger.info(f"Training completed in {training_time:.2f} seconds")
    logger.info(f"Best validation MAE: {best_val_mae:.4f}")
    
    # Save final results
    results = {
        'best_val_mae': best_val_mae,
        'training_time': training_time,
        'model_info': model_info,
        'args': vars(args),
        'history': history
    }
    
    results_path = os.path.join(args.save_dir, "training_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to: {results_path}")

if __name__ == "__main__":
    main()
