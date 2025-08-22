"""
Simple CLIP LSTM Model Training Script
Case 1 + CLIP + LSTM 모델 훈련
"""

import os
import json
import argparse
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.simple_clip_lstm_model import create_simple_clip_lstm_model
from data.mobile_dataset import MobileVLADataset

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('simple_clip_lstm_training.log')
    ]
)
logger = logging.getLogger(__name__)

def train_simple_clip_lstm(args):
    """Simple CLIP LSTM 모델 훈련"""
    
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🚀 디바이스: {device}")
    
    # 데이터셋 로드
    logger.info("📊 데이터셋 로드 중...")
    dataset = MobileVLADataset(
        data_dir=args.data_path,
        sequence_length=18,
        image_size=(224, 224),
        normalize_actions=True
    )
    
    # 훈련/검증 분할
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # 데이터 로더 생성
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True
    )
    
    logger.info("✅ 데이터셋 로드 완료:")
    logger.info(f"   - 훈련셋: {len(train_dataset)} 샘플")
    logger.info(f"   - 검증셋: {len(val_dataset)} 샘플")
    logger.info(f"   - 배치 크기: {args.batch_size}")
    
    # 모델 생성
    logger.info("🤖 모델 생성 중...")
    model, trainer = create_simple_clip_lstm_model()
    model = model.to(device)
    
    # 모델 정보 출력
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"✅ 모델 생성 완료:")
    logger.info(f"   - 총 파라미터 수: {total_params:,}")
    logger.info(f"   - 훈련 가능 파라미터 수: {trainable_params:,}")
    logger.info(f"   - 훈련 비율: {trainable_params/total_params*100:.1f}%")
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 훈련 기록
    train_losses = []
    train_maes = []
    val_losses = []
    val_maes = []
    best_val_mae = float('inf')
    
    # 훈련 루프
    logger.info("🏋️ 훈련 시작...")
    for epoch in range(args.num_epochs):
        # 훈련
        model.train()
        epoch_train_loss = 0.0
        epoch_train_mae = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Train]")
        for batch_idx, batch in enumerate(train_pbar):
            # 훈련 스텝
            loss, mae = trainer.train_step(batch)
            
            epoch_train_loss += loss
            epoch_train_mae += mae
            
            # 진행률 업데이트
            train_pbar.set_postfix({
                'Loss': f'{loss:.4f}',
                'MAE': f'{mae:.4f}'
            })
        
        # 평균 계산
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_train_mae = epoch_train_mae / len(train_loader)
        
        # 검증
        model.eval()
        epoch_val_loss = 0.0
        epoch_val_mae = 0.0
        
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Val]")
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_pbar):
                # 검증 스텝
                loss, mae = trainer.validate_step(batch)
                
                epoch_val_loss += loss
                epoch_val_mae += mae
                
                # 진행률 업데이트
                val_pbar.set_postfix({
                    'Loss': f'{loss:.4f}',
                    'MAE': f'{mae:.4f}'
                })
        
        # 평균 계산
        avg_val_loss = epoch_val_loss / len(val_loader)
        avg_val_mae = epoch_val_mae / len(val_loader)
        
        # 기록 저장
        train_losses.append(avg_train_loss)
        train_maes.append(avg_train_mae)
        val_losses.append(avg_val_loss)
        val_maes.append(avg_val_mae)
        
        # 로그 출력
        logger.info(f"📊 Epoch {epoch+1}/{args.num_epochs}:")
        logger.info(f"   - Train Loss: {avg_train_loss:.4f}")
        logger.info(f"   - Train MAE: {avg_train_mae:.4f}")
        logger.info(f"   - Val Loss: {avg_val_loss:.4f}")
        logger.info(f"   - Val MAE: {avg_val_mae:.4f}")
        
        # 최고 성능 모델 저장 (MAE 기준)
        if avg_val_mae < best_val_mae:
            best_val_mae = avg_val_mae
            best_model_path = os.path.join(args.output_dir, 'best_simple_clip_lstm_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'val_mae': best_val_mae,
                'args': args
            }, best_model_path)
            logger.info(f"💾 최고 성능 모델 저장: {best_model_path}")
        
        # 주기적 체크포인트 저장
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(args.output_dir, f'simple_clip_lstm_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'val_mae': avg_val_mae,
                'args': args
            }, checkpoint_path)
            logger.info(f"💾 체크포인트 저장: {checkpoint_path}")
    
    # 최종 모델 저장
    final_model_path = os.path.join(args.output_dir, 'final_simple_clip_lstm_model.pth')
    torch.save({
        'epoch': args.num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'val_mae': avg_val_mae,
        'args': args
    }, final_model_path)
    
    # 훈련 결과 저장
    results = {
        'train_losses': train_losses,
        'train_maes': train_maes,
        'val_losses': val_losses,
        'val_maes': val_maes,
        'best_val_mae': best_val_mae,
        'final_val_mae': avg_val_mae,
        'args': vars(args)
    }
    
    results_path = os.path.join(args.output_dir, 'simple_clip_lstm_training_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # 훈련 곡선 플롯
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_maes, label='Train MAE')
    plt.plot(val_maes, label='Val MAE')
    plt.title('Training and Validation MAE (2D Navigation)')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, 'simple_clip_lstm_training_curves.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("🎉 훈련 완료!")
    logger.info(f"   - 최고 검증 MAE: {best_val_mae:.4f}")
    logger.info(f"   - 최종 검증 MAE: {avg_val_mae:.4f}")
    logger.info(f"   - 결과 저장: {results_path}")
    logger.info(f"   - 플롯 저장: {plot_path}")
    logger.info(f"   - 최고 모델: {best_model_path}")
    logger.info(f"   - 최종 모델: {final_model_path}")

def main():
    parser = argparse.ArgumentParser(description="Simple CLIP LSTM Model Training (Case 1 + CLIP)")
    
    # 데이터 관련 인자
    parser.add_argument('--data_path', type=str, required=True,
                       help='데이터셋 경로')
    
    # 모델 관련 인자
    parser.add_argument('--batch_size', type=int, default=2,
                       help='배치 크기 (기본값: 2)')
    parser.add_argument('--num_epochs', type=int, default=5,
                       help='훈련 에포크 수 (기본값: 5)')
    
    # 출력 관련 인자
    parser.add_argument('--output_dir', type=str, default='simple_clip_lstm_results',
                       help='결과 저장 디렉토리 (기본값: simple_clip_lstm_results)')
    parser.add_argument('--save_interval', type=int, default=1,
                       help='체크포인트 저장 간격 (기본값: 1)')
    
    args = parser.parse_args()
    
    # 훈련 실행
    train_simple_clip_lstm(args)

if __name__ == "__main__":
    main()
