#!/usr/bin/env python3
"""
🎯 앙상블 모델 학습 스크립트
LSTM + MLP Action Head 조합 모델 학습
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import json
import os
from pathlib import Path
import time
from datetime import datetime
import sys

# 경로 설정
sys.path.append('/home/billy/25-1kp/vla/Robo+/Mobile_VLA')
from core.data_core.mobile_vla_dataset import MobileVLADataset
from ensemble_action_head_model import create_ensemble_model

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_ensemble_model(
    epochs: int = 5,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_dir: str = "ensemble_action_head_results"
):
    """앙상블 모델 학습"""
    
    logger.info("🚀 앙상블 Action Head 모델 학습 시작")
    logger.info(f"Device: {device}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Learning rate: {learning_rate}")
    
    # 결과 저장 디렉토리 생성
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    # 앙상블 모델 생성
    logger.info("앙상블 모델 생성 중...")
    ensemble_model = create_ensemble_model(
        lstm_model_path="enhanced_kosmos2_clip_hybrid_with_normalization_results/best_enhanced_kosmos2_clip_hybrid_with_mobile_normalization.pth",
        mlp_model_path="Robo+/Mobile_VLA/results/mobile_vla_epoch_3.pt",
        action_dim=2,
        fusion_method="weighted"
    )
    
    ensemble_model = ensemble_model.to(device)
    
    # 데이터셋 로드
    logger.info("데이터셋 로드 중...")
    dataset = MobileVLADataset(
        data_dir="ROS_action/mobile_vla_dataset",
        max_episodes=72,
        image_size=224,
        action_dim=2
    )
    
    # Train/Val split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    
    # 옵티마이저 및 손실 함수
    optimizer = optim.AdamW(ensemble_model.parameters(), lr=learning_rate, weight_decay=0.01)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # 학습 히스토리
    history = {
        "train_loss": [],
        "train_mae": [],
        "val_loss": [],
        "val_mae": [],
        "learning_rate": [],
        "ensemble_weights": []
    }
    
    best_val_loss = float('inf')
    best_model_path = None
    
    # 학습 루프
    for epoch in range(epochs):
        logger.info(f"\n📊 Epoch {epoch+1}/{epochs}")
        
        # Training
        ensemble_model.train()
        train_loss = 0.0
        train_mae = 0.0
        train_samples = 0
        
        for batch_idx, (images, actions) in enumerate(train_loader):
            images = images.to(device)
            actions = actions.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            predicted_actions = ensemble_model(images)
            loss = criterion(predicted_actions, actions)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # 통계 업데이트
            train_loss += loss.item() * images.size(0)
            mae = torch.mean(torch.abs(predicted_actions - actions)).item()
            train_mae += mae * images.size(0)
            train_samples += images.size(0)
            
            if batch_idx % 10 == 0:
                logger.info(f"  Batch {batch_idx}/{len(train_loader)}: Loss={loss.item():.4f}, MAE={mae:.4f}")
        
        # Validation
        ensemble_model.eval()
        val_loss = 0.0
        val_mae = 0.0
        val_samples = 0
        
        with torch.no_grad():
            for images, actions in val_loader:
                images = images.to(device)
                actions = actions.to(device)
                
                predicted_actions = ensemble_model(images)
                loss = criterion(predicted_actions, actions)
                
                val_loss += loss.item() * images.size(0)
                mae = torch.mean(torch.abs(predicted_actions - actions)).item()
                val_mae += mae * images.size(0)
                val_samples += images.size(0)
        
        # 평균 계산
        avg_train_loss = train_loss / train_samples
        avg_train_mae = train_mae / train_samples
        avg_val_loss = val_loss / val_samples
        avg_val_mae = val_mae / val_samples
        current_lr = optimizer.param_groups[0]['lr']
        
        # 앙상블 가중치 추출
        if hasattr(ensemble_model, 'ensemble_weights'):
            ensemble_weights = ensemble_model.ensemble_weights.detach().cpu().numpy().tolist()
        else:
            ensemble_weights = [0.5, 0.5]  # 기본값
        
        # 히스토리 업데이트
        history["train_loss"].append(avg_train_loss)
        history["train_mae"].append(avg_train_mae)
        history["val_loss"].append(avg_val_loss)
        history["val_mae"].append(avg_val_mae)
        history["learning_rate"].append(current_lr)
        history["ensemble_weights"].append(ensemble_weights)
        
        logger.info(f"Train Loss: {avg_train_loss:.4f}, Train MAE: {avg_train_mae:.4f}")
        logger.info(f"Val Loss: {avg_val_loss:.4f}, Val MAE: {avg_val_mae:.4f}")
        logger.info(f"Learning Rate: {current_lr:.6f}")
        logger.info(f"Ensemble Weights: LSTM={ensemble_weights[0]:.3f}, MLP={ensemble_weights[1]:.3f}")
        
        # 최고 모델 저장
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = save_path / f"best_ensemble_model_epoch_{epoch+1}.pth"
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': ensemble_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'train_mae': avg_train_mae,
                'val_loss': avg_val_loss,
                'val_mae': avg_val_mae,
                'ensemble_weights': ensemble_weights,
                'model_info': ensemble_model.get_model_info()
            }, best_model_path)
            
            logger.info(f"✅ 새로운 최고 모델 저장: {best_model_path}")
        
        # 에포크별 모델 저장
        epoch_model_path = save_path / f"ensemble_model_epoch_{epoch+1}.pth"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': ensemble_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'train_mae': avg_train_mae,
            'val_loss': avg_val_loss,
            'val_mae': avg_val_mae,
            'ensemble_weights': ensemble_weights,
            'model_info': ensemble_model.get_model_info()
        }, epoch_model_path)
        
        scheduler.step()
    
    # 최종 모델 저장
    final_model_path = save_path / "final_ensemble_model.pth"
    torch.save({
        'epoch': epochs,
        'model_state_dict': ensemble_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': avg_train_loss,
        'train_mae': avg_train_mae,
        'val_loss': avg_val_loss,
        'val_mae': avg_val_mae,
        'ensemble_weights': ensemble_weights,
        'model_info': ensemble_model.get_model_info()
    }, final_model_path)
    
    # 학습 히스토리 저장
    history_path = save_path / "training_history_ensemble.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # 모델 정보 저장
    model_info_path = save_path / "model_info.json"
    with open(model_info_path, 'w') as f:
        json.dump(ensemble_model.get_model_info(), f, indent=2)
    
    logger.info(f"\n🎉 앙상블 모델 학습 완료!")
    logger.info(f"최고 Val Loss: {best_val_loss:.4f}")
    logger.info(f"최고 모델: {best_model_path}")
    logger.info(f"최종 앙상블 가중치: LSTM={ensemble_weights[0]:.3f}, MLP={ensemble_weights[1]:.3f}")
    logger.info(f"결과 저장 위치: {save_path}")
    
    return {
        "best_val_loss": best_val_loss,
        "best_model_path": str(best_model_path),
        "final_model_path": str(final_model_path),
        "final_ensemble_weights": ensemble_weights,
        "history": history
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="앙상블 Action Head 모델 학습")
    parser.add_argument("--epochs", type=int, default=5, help="학습 에포크 수")
    parser.add_argument("--batch_size", type=int, default=4, help="배치 크기")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="학습률")
    parser.add_argument("--device", type=str, default="cuda", help="사용할 디바이스")
    parser.add_argument("--save_dir", type=str, default="ensemble_action_head_results", help="결과 저장 디렉토리")
    
    args = parser.parse_args()
    
    # 학습 실행
    results = train_ensemble_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        save_dir=args.save_dir
    )
    
    print(f"\n📊 앙상블 모델 최종 결과:")
    print(f"Best Val Loss: {results['best_val_loss']:.4f}")
    print(f"Best Model: {results['best_model_path']}")
    print(f"Final Ensemble Weights: LSTM={results['final_ensemble_weights'][0]:.3f}, MLP={results['final_ensemble_weights'][1]:.3f}")
