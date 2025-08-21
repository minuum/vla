#!/usr/bin/env python3
"""
Case 3: 중기 적용 (Medium-term Optimization)
고급 RoboVLMs 기능 + 멀티모달 융합 모델 훈련
"""
import os
import sys
import argparse
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoModel
from tqdm import tqdm
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import h5py
from PIL import Image

# 프로젝트 루트 추가
sys.path.append(str(Path(__file__).parent.parent.parent))

from advanced_multimodal_model import AdvancedMultimodalModel, AdvancedMultimodalTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedDataset:
    """고급 데이터셋 - 멀티모달 융합을 위한 데이터 로더"""
    
    def __init__(self, data_path, processor, frame_selection='first'):
        self.data_path = data_path
        self.processor = processor
        self.frame_selection = frame_selection
        self.data = self._load_data()
        logger.info(f"✅ Advanced Dataset 초기화 완료:")
        logger.info(f"   - 데이터 경로: {data_path}")
        logger.info(f"   - 샘플 수: {len(self.data)}")

    def _load_data(self):
        """H5 파일들에서 데이터 로드"""
        data = []
        data_path = Path(self.data_path)
        h5_files = list(data_path.glob("*.h5"))
        logger.info(f"📁 H5 파일 수: {len(h5_files)}")

        for h5_file in h5_files:
            try:
                with h5py.File(h5_file, 'r') as f:
                    if 'images' in f and 'actions' in f:
                        images = f['images'][:]
                        actions = f['actions'][:]
                        
                        for frame_idx in range(len(images)):
                            if self.frame_selection == 'first' and frame_idx != 0:
                                continue
                            elif self.frame_selection == 'random' and frame_idx != np.random.randint(0, len(images) - 1):
                                continue
                            
                            data.append({
                                'image': images[frame_idx],
                                'action': actions[frame_idx][:2],  # 2D 액션만
                                'episode_id': len(data),
                                'frame_id': frame_idx
                            })
                            if self.frame_selection == 'first':
                                break
            except Exception as e:
                logger.error(f"❌ {h5_file} 로드 오류: {e}")
                continue
        
        logger.info(f"📊 로드된 샘플 수: {len(data)}")
        return data

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.fromarray(item['image']).convert('RGB')
        action = torch.tensor(item['action'], dtype=torch.float32)
        
        # 고급 시나리오 명령어 생성
        scenario = self._extract_scenario_from_filename(item.get('filename', ''))
        text = f"Navigate the robot to {scenario} location."
        
        return {
            'image': image,
            'action': action,
            'text': text,
            'episode_id': item['episode_id']
        }

    def _extract_scenario_from_filename(self, filename):
        """파일명에서 시나리오 추출"""
        if 'hori' in filename:
            return 'horizontal'
        elif 'vert' in filename:
            return 'vertical'
        elif 'close' in filename:
            return 'close'
        elif 'medium' in filename:
            return 'medium'
        elif 'far' in filename:
            return 'far'
        else:
            return 'target'

    def __len__(self):
        return len(self.data)

def custom_collate_fn(batch):
    """PIL 이미지를 처리하는 커스텀 collate 함수"""
    images = [item['image'] for item in batch]
    actions = torch.stack([item['action'] for item in batch])
    texts = [item['text'] for item in batch]
    episode_ids = [item['episode_id'] for item in batch]
    
    return {
        'image': images,  # PIL 이미지 리스트
        'action': actions,
        'text': texts,
        'episode_id': episode_ids
    }

def create_advanced_data_loaders(data_path, processor, batch_size=2,
                                train_split=0.7, val_split=0.15, test_split=0.15):
    """고급 데이터 로더 생성"""
    full_dataset = AdvancedDataset(
        data_path=data_path,
        processor=processor,
        frame_selection='first'
    )
    
    total_size = len(full_dataset)
    train_size = int(total_size * train_split)
    val_size = int(total_size * val_split)
    test_size = total_size - train_size - val_size
    
    logger.info(f"📊 데이터셋 분할:")
    logger.info(f"   - 전체: {total_size}")
    logger.info(f"   - 훈련: {train_size}")
    logger.info(f"   - 검증: {val_size}")
    logger.info(f"   - 테스트: {test_size}")
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    logger.info(f"✅ Advanced Data Loaders 생성 완료")
    return train_loader, val_loader, test_loader

def train_advanced_model(data_path, output_dir, num_epochs=50, batch_size=2,
                        learning_rate=3e-5, weight_decay=1e-4, patience=5):
    """고급 멀티모달 모델 훈련"""
    
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🚀 Case 3 훈련 시작 - 디바이스: {device}")
    
    # Kosmos2 프로세서 로드
    logger.info("📥 Kosmos2 프로세서 로드 중...")
    processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
    
    # 고급 데이터 로더 생성
    logger.info("📊 고급 데이터 로더 생성 중...")
    train_loader, val_loader, test_loader = create_advanced_data_loaders(
        data_path=data_path,
        processor=processor,
        batch_size=batch_size
    )
    
    # 모델 및 훈련기 생성
    logger.info("🤖 고급 모델 및 훈련기 생성 중...")
    model = AdvancedMultimodalModel(
        processor=processor,
        vision_dim=1024,
        language_dim=2048,  # Kosmos2 text 모델 출력 차원
        action_dim=2,
        hidden_dim=512,  # 더 큰 hidden_dim
        dropout=0.3,
        use_hierarchical_planning=True
    ).to(device)
    
    trainer = AdvancedMultimodalTrainer(
        model=model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        device=device
    )
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 훈련 설정
    logger.info("🎯 훈련 설정:")
    logger.info(f"   - 에포크: {num_epochs}")
    logger.info(f"   - 배치 크기: {batch_size}")
    logger.info(f"   - 학습률: {learning_rate}")
    logger.info(f"   - Weight Decay: {weight_decay}")
    logger.info(f"   - 조기 종료 인내심: {patience}")
    
    # 훈련 루프
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        logger.info(f"\n📈 Epoch {epoch+1}/{num_epochs}")
        
        # 훈련 단계
        model.train()
        train_losses = []
        train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        
        for batch in train_pbar:
            try:
                loss = trainer.train_step(batch)
                train_losses.append(loss)
                train_pbar.set_postfix({'loss': f'{loss:.4f}'})
            except Exception as e:
                logger.error(f"❌ 훈련 배치 오류: {e}")
                continue
        
        avg_train_loss = np.mean(train_losses)
        
        # 검증 단계
        model.eval()
        val_losses = []
        val_pbar = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}")
        
        with torch.no_grad():
            for batch in val_pbar:
                try:
                    loss, mae = trainer.validate_step(batch)
                    val_losses.append(loss)
                    val_pbar.set_postfix({'loss': f'{loss:.4f}', 'mae': f'{mae:.4f}'})
                except Exception as e:
                    logger.error(f"❌ 검증 배치 오류: {e}")
                    continue
        
        avg_val_loss = np.mean(val_losses)
        
        # 로깅
        logger.info(f"📊 Epoch {epoch+1} 결과:")
        logger.info(f"   - 훈련 손실: {avg_train_loss:.4f}")
        logger.info(f"   - 검증 손실: {avg_val_loss:.4f}")
        
        # 모델 저장
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # 최고 모델 저장
            model_path = os.path.join(output_dir, f"best_model_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'best_val_loss': best_val_loss
            }, model_path)
            logger.info(f"💾 최고 모델 저장: {model_path}")
        else:
            patience_counter += 1
            logger.info(f"⏳ 조기 종료 카운터: {patience_counter}/{patience}")
        
        # 조기 종료 체크
        if patience_counter >= patience:
            logger.info(f"🛑 조기 종료! {patience} 에포크 동안 개선 없음")
            break
        
        # 중간 모델 저장 (10 에포크마다)
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss
            }, checkpoint_path)
            logger.info(f"💾 중간 체크포인트 저장: {checkpoint_path}")
    
    # 최종 모델 저장
    final_model_path = os.path.join(output_dir, "final_model.pth")
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'train_loss': avg_train_loss,
        'val_loss': avg_val_loss,
        'best_val_loss': best_val_loss
    }, final_model_path)
    logger.info(f"💾 최종 모델 저장: {final_model_path}")
    
    # 훈련 결과 저장
    results = {
        'final_epoch': epoch + 1,
        'best_val_loss': best_val_loss,
        'final_train_loss': avg_train_loss,
        'final_val_loss': avg_val_loss,
        'training_completed': True,
        'early_stopped': patience_counter >= patience
    }
    
    results_path = os.path.join(output_dir, "training_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"📊 훈련 결과 저장: {results_path}")
    
    logger.info("🎉 Case 3 훈련 완료!")
    return model, trainer

def main():
    parser = argparse.ArgumentParser(description="Case 3: 고급 멀티모달 모델 훈련")
    parser.add_argument("--data_path", type=str, required=True,
                       help="데이터셋 경로")
    parser.add_argument("--output_dir", type=str, default="case3_results",
                       help="출력 디렉토리")
    parser.add_argument("--num_epochs", type=int, default=50,
                       help="훈련 에포크 수")
    parser.add_argument("--batch_size", type=int, default=2,
                       help="배치 크기")
    parser.add_argument("--learning_rate", type=float, default=3e-5,
                       help="학습률")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                       help="Weight decay")
    parser.add_argument("--patience", type=int, default=5,
                       help="조기 종료 인내심")
    
    args = parser.parse_args()
    
    # 훈련 실행
    model, trainer = train_advanced_model(
        data_path=args.data_path,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience
    )

if __name__ == "__main__":
    main()
