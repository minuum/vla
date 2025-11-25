#!/usr/bin/env python3
"""
Mobile VLA Training Script for RoboVLMs Framework
Full Fine-tuning with LSTM Policy Head (CALVIN Standard)
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

# Add RoboVLMs to path
sys.path.insert(0, str(Path(__file__).parent))

from robovlms.data.mobile_vla_dataset import get_mobile_vla_dataset
from robovlms.train.base_trainer import BaseTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def setup_trainer(config: Dict) -> pl.Trainer:
    """Setup PyTorch Lightning Trainer (CALVIN Standard)"""
    trainer_config = config.get('trainer', {})
    
    # Callbacks
    callbacks = []
    
    # Model checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(config['output_root'], config['task_name']),
        filename='mobile_vla-{epoch:02d}-{val_loss:.4f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min',
        save_last=True,
    )
    callbacks.append(checkpoint_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    # Logger
    tb_logger = TensorBoardLogger(
        save_dir=config['log_root'],
        name=config['task_name'],
    )
    
    # Create trainer (CALVIN standard settings)
    trainer = pl.Trainer(
        accelerator=trainer_config.get('accelerator', 'gpu'),
        devices=trainer_config.get('devices', 1),
        strategy=trainer_config.get('strategy', 'auto'),
        precision=trainer_config.get('precision', 'bf16'),
        max_epochs=trainer_config.get('max_epochs', 5),
        max_steps=trainer_config.get('max_steps', -1),
        val_check_interval=trainer_config.get('val_check_interval', 0.5),
        check_val_every_n_epoch=trainer_config.get('check_val_every_n_epoch', 1),
        log_every_n_steps=trainer_config.get('log_every_n_steps', 10),
        gradient_clip_val=trainer_config.get('gradient_clip_val', 1.0),
        accumulate_grad_batches=trainer_config.get('accumulate_grad_batches', 4),
        callbacks=callbacks,
        logger=tb_logger,
        enable_checkpointing=True,
        default_root_dir=trainer_config.get('default_root_dir', 'runs/mobile_vla'),
    )
    
    return trainer


def setup_model(config: Dict) -> BaseTrainer:
    """Setup RoboVLMs model with Mobile VLA configuration"""
    
    # Create model
    model = BaseTrainer(config)
    
    logger.info(f"Model created: {config['robovlm_name']}")
    logger.info(f"Action space: {config['act_head']['action_space']}")
    logger.info(f"Action dim: {config['act_head']['action_dim']}")
    logger.info(f"LSTM layers: {config['act_head'].get('num_layers', 4)}")
    logger.info(f"Window size: {config['window_size']}")
    logger.info(f"Predict next: {config['fwd_pred_next_n']}")
    
    # Print trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Trainable ratio: {trainable_params/total_params*100:.2f}%")
    
    return model


def setup_dataloaders(config: Dict):
    """Setup train and validation dataloaders"""
    from torch.utils.data import DataLoader
    
    # Create datasets
    train_dataset, val_dataset = get_mobile_vla_dataset(config)
    
    logger.info(f"Train dataset: {len(train_dataset)} samples")
    logger.info(f"Val dataset: {len(val_dataset)} samples")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        collate_fn=train_dataset.collater,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        collate_fn=val_dataset.collater,
        pin_memory=True,
    )
    
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description='Train Mobile VLA with RoboVLMs')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/mobile_vla/train_mobile_vla_full_ft.json',
        help='Path to config file'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run in test mode (1 epoch, small dataset)'
    )
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    logger.info(f"Loaded config from {args.config}")
    
    # Test mode
    if args.test:
        logger.info("Running in TEST mode")
        config['trainer']['max_epochs'] = 1
        config['trainer']['max_steps'] = 10
        config['batch_size'] = 2
    
    # Resume from checkpoint
    if args.resume:
        config['resume'] = args.resume
        logger.info(f"Resuming from checkpoint: {args.resume}")
    
    # Create output directories
    os.makedirs(config['output_root'], exist_ok=True)
    os.makedirs(config['log_root'], exist_ok=True)
    
    # Setup
    logger.info("Setting up model...")
    model = setup_model(config)
    
    logger.info("Setting up dataloaders...")
    train_loader, val_loader = setup_dataloaders(config)
    
    logger.info("Setting up trainer...")
    trainer = setup_trainer(config)
    
    # Train
    logger.info("Starting training...")
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=args.resume
    )
    
    logger.info("Training completed!")
    logger.info(f"Best model saved to: {trainer.checkpoint_callback.best_model_path}")


if __name__ == '__main__':
    main()

