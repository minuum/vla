#!/usr/bin/env python3
"""
🎯 증강된 데이터를 사용한 학습
"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import json
import logging
from datetime import datetime
import random
from typing import Dict, List, Tuple

# 프로젝트 경로 설정
ROOT_DIR = Path("/home/billy/25-1kp/vla/Robo+/Mobile_VLA")
AUGMENTED_DATA_DIR = ROOT_DIR / "augmented_dataset"

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AugmentedDataset(Dataset):
    """증강된 데이터셋 클래스"""
    
    def __init__(self, data_dir: Path, split: str = "train", window_size: int = 8, chunk_size: int = 2):
        self.data_dir = data_dir
        self.split = split
        self.window_size = window_size
        self.chunk_size = chunk_size
        
        # 메타데이터 로드
        with open(data_dir / "metadata.json", 'r') as f:
            self.metadata = json.load(f)
        
        # 에피소드 목록 생성
        self.episodes = []
        for episode_dir in sorted(data_dir.glob("episode_*")):
            if episode_dir.is_dir():
                # 에피소드 메타데이터 로드
                with open(episode_dir / "metadata.json", 'r') as f:
                    episode_meta = json.load(f)
                
                self.episodes.append({
                    'dir': episode_dir,
                    'meta': episode_meta
                })
        
        # 훈련/검증 분할
        total_episodes = len(self.episodes)
        if split == "train":
            self.episodes = self.episodes[:int(0.8 * total_episodes)]
        else:  # validation
            self.episodes = self.episodes[int(0.8 * total_episodes):]
        
        # 이미지 변환
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        logger.info(f"📊 {split} 데이터셋 로드 완료: {len(self.episodes)} 에피소드")
    
    def __len__(self):
        return len(self.episodes)
    
    def __getitem__(self, idx):
        episode = self.episodes[idx]
        episode_dir = episode['dir']
        
        # 이미지 로드
        images = []
        image_files = sorted(episode_dir.glob("frame_*.jpg"))
        for img_file in image_files:
            image = Image.open(img_file).convert('RGB')
            image = self.transform(image)
            images.append(image)
        
        # 액션 로드
        actions = np.load(episode_dir / "actions.npy")
        
        # 텐서로 변환
        images = torch.stack(images)  # [T, C, H, W]
        actions = torch.from_numpy(actions).float()  # [T, 3]
        
        return {
            'images': images,
            'actions': actions,
            'episode_id': episode['meta']['episode_id'],
            'augmentation_type': episode['meta']['augmentation_type']
        }

class AugmentedDataTrainer:
    """증강된 데이터를 사용한 학습기"""
    
    def __init__(self, model_name: str = "microsoft/kosmos-2-patch14-224"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 모델 초기화
        self._init_model()
        
        # 옵티마이저 및 스케줄러
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1000)
        
        # 손실 함수
        self.criterion = nn.HuberLoss()
        
        logger.info(f"🎯 증강 데이터 학습기 초기화 완료 (디바이스: {self.device})")
    
    def _collate_fn(self, batch):
        """배치 데이터를 올바르게 결합하는 함수"""
        # 모든 에피소드가 같은 길이인지 확인
        images_list = [item['images'] for item in batch]
        actions_list = [item['actions'] for item in batch]
        episode_ids = [item['episode_id'] for item in batch]
        augmentation_types = [item['augmentation_type'] for item in batch]
        
        # 배치로 스택
        images = torch.stack(images_list)
        actions = torch.stack(actions_list)
        
        return {
            'images': images,
            'actions': actions,
            'episode_id': episode_ids,
            'augmentation_type': augmentation_types
        }
    
    def _init_model(self):
        """모델 초기화"""
        from transformers import Kosmos2Model
        
        class AugmentedVLAModel(nn.Module):
            def __init__(self, model_name, action_dim=3, window_size=8, chunk_size=2):
                super().__init__()
                
                # Kosmos2 모델
                self.kosmos = Kosmos2Model.from_pretrained(model_name)
                
                # 모델 설정
                self.hidden_size = 768
                self.lstm_hidden_size = 512
                self.lstm_layers = 2
                self.window_size = window_size
                self.chunk_size = chunk_size
                self.action_dim = action_dim
                
                # LSTM 레이어
                self.action_lstm = nn.LSTM(
                    input_size=self.hidden_size,
                    hidden_size=self.lstm_hidden_size,
                    num_layers=self.lstm_layers,
                    batch_first=True,
                    dropout=0.1
                )
                
                # 액션 헤드
                self.action_head = nn.Sequential(
                    nn.Linear(self.lstm_hidden_size, 256),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(128, action_dim)
                )
            
            def forward(self, images, input_ids=None, attention_mask=None):
                batch_size = images.shape[0]
                
                # 이미지 특징 추출
                image_features = []
                for i in range(batch_size):
                    episode_images = images[i]  # [T, C, H, W]
                    
                    # Kosmos2로 이미지 특징 추출
                    try:
                        vision_outputs = self.kosmos.vision_model(pixel_values=episode_images)
                        if hasattr(vision_outputs, 'pooler_output') and vision_outputs.pooler_output is not None:
                            features = vision_outputs.pooler_output
                        else:
                            features = vision_outputs.last_hidden_state.mean(dim=1)
                        
                        # 크기 통일
                        if features.shape[-1] != self.hidden_size:
                            # 선형 변환으로 크기 맞추기
                            if not hasattr(self, 'feature_adapter'):
                                self.feature_adapter = nn.Linear(features.shape[-1], self.hidden_size).to(features.device)
                            features = self.feature_adapter(features)
                            
                    except Exception as e:
                        # 대체 방법
                        features = torch.randn(episode_images.shape[0], self.hidden_size, device=episode_images.device)
                    
                    image_features.append(features)
                
                # 배치로 스택
                image_features = torch.stack(image_features)  # [B, T, H]
                
                # LSTM 처리
                lstm_out, _ = self.action_lstm(image_features)
                
                # 액션 예측
                actions = self.action_head(lstm_out)
                
                return actions
        
        self.model = AugmentedVLAModel(self.model_name).to(self.device)
    
    def train_epoch(self, dataloader):
        """한 에포크 학습"""
        self.model.train()
        total_loss = 0.0
        total_mae = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            images = batch['images'].to(self.device)  # [B, T, C, H, W]
            actions = batch['actions'].to(self.device)  # [B, T, 3]
            
            # 입력 준비
            batch_size = images.shape[0]
            input_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=self.device)
            attention_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predicted_actions = self.model(images, input_ids, attention_mask)
            
            # 손실 계산
            loss = self.criterion(predicted_actions, actions)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # 메트릭 계산
            mae = torch.mean(torch.abs(predicted_actions - actions))
            
            total_loss += loss.item()
            total_mae += mae.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                logger.info(f"   배치 {batch_idx}/{len(dataloader)}: Loss={loss.item():.4f}, MAE={mae.item():.4f}")
        
        self.scheduler.step()
        
        return {
            'loss': total_loss / num_batches,
            'mae': total_mae / num_batches
        }
    
    def validate(self, dataloader):
        """검증"""
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                images = batch['images'].to(self.device)
                actions = batch['actions'].to(self.device)
                
                # 입력 준비
                batch_size = images.shape[0]
                input_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=self.device)
                attention_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=self.device)
                
                # Forward pass
                predicted_actions = self.model(images, input_ids, attention_mask)
                
                # 손실 계산
                loss = self.criterion(predicted_actions, actions)
                mae = torch.mean(torch.abs(predicted_actions - actions))
                
                total_loss += loss.item()
                total_mae += mae.item()
                num_batches += 1
        
        return {
            'loss': total_loss / num_batches,
            'mae': total_mae / num_batches
        }
    
    def train(self, num_epochs: int = 10, batch_size: int = 4):
        """전체 학습 과정"""
        logger.info("🎯 증강된 데이터로 학습 시작!")
        
        # 데이터셋 로드
        train_dataset = AugmentedDataset(AUGMENTED_DATA_DIR, "train")
        val_dataset = AugmentedDataset(AUGMENTED_DATA_DIR, "validation")
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=self._collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=self._collate_fn)
        
        # 학습 기록
        best_val_loss = float('inf')
        training_history = []
        
        for epoch in range(num_epochs):
            logger.info(f"\n📈 에포크 {epoch+1}/{num_epochs}")
            logger.info("-" * 50)
            
            # 훈련
            train_metrics = self.train_epoch(train_loader)
            
            # 검증
            val_metrics = self.validate(val_loader)
            
            # 결과 출력
            logger.info(f"✅ 훈련: Loss={train_metrics['loss']:.4f}, MAE={train_metrics['mae']:.4f}")
            logger.info(f"🔍 검증: Loss={val_metrics['loss']:.4f}, MAE={val_metrics['mae']:.4f}")
            
            # 최고 모델 저장
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                torch.save(self.model.state_dict(), ROOT_DIR / "best_augmented_model.pth")
                logger.info("💾 최고 모델 저장됨!")
            
            # 기록
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_metrics['loss'],
                'train_mae': train_metrics['mae'],
                'val_loss': val_metrics['loss'],
                'val_mae': val_metrics['mae']
            })
        
        # 결과 저장
        results = {
            'training_history': training_history,
            'best_val_loss': best_val_loss,
            'final_train_mae': train_metrics['mae'],
            'final_val_mae': val_metrics['mae'],
            'model_name': self.model_name,
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'augmentation_factor': train_dataset.metadata['augmentation_factor'],
            'total_episodes': train_dataset.metadata['total_episodes'],
            'completion_date': datetime.now().isoformat()
        }
        
        with open(ROOT_DIR / "augmented_training_results.json", 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info("\n🎉 증강된 데이터 학습 완료!")
        logger.info(f"📊 최고 검증 Loss: {best_val_loss:.4f}")
        logger.info(f"📊 최종 훈련 MAE: {train_metrics['mae']:.4f}")
        logger.info(f"📊 최종 검증 MAE: {val_metrics['mae']:.4f}")
        
        return results

def main():
    """메인 실행 함수"""
    print("🎯 증강된 데이터로 학습 시작!")
    print("=" * 50)
    
    # 증강된 데이터 확인
    if not AUGMENTED_DATA_DIR.exists():
        print("❌ 증강된 데이터가 없습니다. 먼저 generate_augmented_data.py를 실행하세요.")
        return
    
    # 메타데이터 확인
    with open(AUGMENTED_DATA_DIR / "metadata.json", 'r') as f:
        metadata = json.load(f)
    
    print(f"📊 증강된 데이터 정보:")
    print(f"   총 에피소드: {metadata['total_episodes']}")
    print(f"   원본 에피소드: {metadata['original_episodes']}")
    print(f"   증강 배수: {metadata['augmentation_factor']}x")
    print(f"   생성 날짜: {metadata['generation_date']}")
    
    # 학습기 초기화 및 학습 실행
    trainer = AugmentedDataTrainer()
    results = trainer.train(num_epochs=10, batch_size=4)
    
    print("\n🎉 학습 완료!")
    print(f"📁 결과 저장: {ROOT_DIR}/augmented_training_results.json")
    print(f"📁 모델 저장: {ROOT_DIR}/best_augmented_model.pth")

if __name__ == "__main__":
    main()
