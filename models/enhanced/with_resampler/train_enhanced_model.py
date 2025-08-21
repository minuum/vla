"""
🚀 Train Enhanced 2D Model with Vision Resampler
Vision Resampler를 포함한 향상된 2D 모델 훈련 스크립트
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys
import logging
from pathlib import Path
from transformers import AutoProcessor, AutoModel
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import argparse

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from enhanced_2d_model_complete import Enhanced2DActionModel
from enhanced_dataset import create_enhanced_data_loaders

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def train_enhanced_2d_model(
    model,
    train_loader,
    val_loader,
    num_epochs=15,
    learning_rate=1e-4,
    weight_decay=1e-4,
    early_stopping_patience=5,
    device='cuda',
    save_dir='checkpoints'
):
    """Train enhanced 2D action model with Vision Resampler"""
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    model = model.to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs
    )
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': []
    }
    
    logger.info(f"🚀 Enhanced 2D action model training started")
    logger.info(f"   - Epochs: {num_epochs}")
    logger.info(f"   - Learning rate: {learning_rate}")
    logger.info(f"   - Weight decay: {weight_decay}")
    logger.info(f"   - Device: {device}")
    logger.info(f"   - Save directory: {save_dir}")
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        num_batches = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch in train_pbar:
            images = batch['image'].to(device)
            actions = batch['action'].to(device)
            texts = batch['text']
            
            # Forward pass
            predictions = model(images, texts)
            
            # Loss calculation
            action_loss = criterion(predictions, actions)
            
            # Backward pass
            optimizer.zero_grad()
            action_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += action_loss.item()
            num_batches += 1
            
            # Update progress bar
            train_pbar.set_postfix({
                'loss': f'{action_loss.item():.6f}',
                'avg_loss': f'{train_loss/num_batches:.6f}'
            })
        
        # Validation
        model.eval()
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for batch in val_pbar:
                images = batch['image'].to(device)
                actions = batch['action'].to(device)
                texts = batch['text']
                
                # Forward pass
                predictions = model(images, texts)
                
                # Loss calculation
                action_loss = criterion(predictions, actions)
                
                val_loss += action_loss.item()
                val_batches += 1
                
                # Update progress bar
                val_pbar.set_postfix({
                    'loss': f'{action_loss.item():.6f}',
                    'avg_loss': f'{val_loss/val_batches:.6f}'
                })
        
        # Calculate average losses
        avg_train_loss = train_loss / num_batches
        avg_val_loss = val_loss / val_batches
        current_lr = scheduler.get_last_lr()[0]
        
        # Update scheduler
        scheduler.step()
        
        # Save to history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['learning_rate'].append(current_lr)
        
        # Print results
        logger.info(f"Epoch {epoch+1}/{num_epochs}:")
        logger.info(f"  Train - Action Loss: {avg_train_loss:.6f}")
        logger.info(f"  Val   - Action Loss: {avg_val_loss:.6f}")
        logger.info(f"  LR: {current_lr:.2e}")
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Save best model
            best_model_path = os.path.join(save_dir, f'enhanced_2d_model_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': avg_val_loss,
                'config': {
                    'use_vision_resampler': model.use_vision_resampler,
                    'action_dim': model.action_dim,
                    'hidden_dim': model.hidden_dim,
                    'dropout': model.dropout
                }
            }, best_model_path)
            logger.info(f"  ✅ Best model saved (Val Loss: {avg_val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logger.info(f"  🛑 Early stopping (Patience: {early_stopping_patience})")
                break
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(save_dir, f'enhanced_2d_model_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': avg_val_loss,
                'history': history
            }, checkpoint_path)
            logger.info(f"  💾 Checkpoint saved: {checkpoint_path}")
    
    # Save final model
    final_model_path = os.path.join(save_dir, 'enhanced_2d_model_final.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss': avg_val_loss,
        'history': history,
        'config': {
            'use_vision_resampler': model.use_vision_resampler,
            'action_dim': model.action_dim,
            'hidden_dim': model.hidden_dim,
            'dropout': model.dropout
        }
    }, final_model_path)
    
    # Save training history
    history_path = os.path.join(save_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info(f"🎉 Enhanced 2D action model training completed!")
    logger.info(f"   Best validation loss: {best_val_loss:.6f}")
    logger.info(f"   Final model saved: {final_model_path}")
    logger.info(f"   Training history saved: {history_path}")
    
    return model, history

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train Enhanced 2D Model with Vision Resampler')
    parser.add_argument('--data_path', type=str, required=True, help='Path to H5 data directory')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--num_epochs', type=int, default=15, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--early_stopping_patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--frame_selection', type=str, default='random', 
                       choices=['random', 'middle', 'all'], help='Frame selection strategy')
    
    args = parser.parse_args()
    
    # Device setup
    device = args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Load processor
    logger.info("Loading Kosmos2 processor...")
    processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
    
    # Create enhanced model
    logger.info("Creating enhanced 2D action model...")
    model = Enhanced2DActionModel(
        processor=processor,
        vision_dim=1024,
        language_dim=1024,
        action_dim=2,
        hidden_dim=512,
        dropout=0.2,
        use_vision_resampler=True  # Enable vision resampler
    )
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader = create_enhanced_data_loaders(
        data_path=args.data_path,
        processor=processor,
        batch_size=args.batch_size,
        train_split=0.8,
        frame_selection=args.frame_selection,
        use_vision_resampler=True
    )
    
    # Train model
    logger.info("Starting training...")
    trained_model, history = train_enhanced_2d_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        early_stopping_patience=args.early_stopping_patience,
        device=device,
        save_dir=args.save_dir
    )
    
    logger.info("✅ Training completed successfully!")

if __name__ == "__main__":
    main()

