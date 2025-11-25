#!/usr/bin/env python3
"""
Original 72 Episodes CLIP 모델을 증강 데이터로 학습
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, CLIPVisionModel, CLIPTextModel
import numpy as np
import logging
import argparse
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import h5py
from PIL import Image
import torchvision.transforms as transforms

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AugmentedDataset(Dataset):
    """증강 데이터셋"""
    
    def __init__(self, data_dir, processor, transform=None):
        self.data_dir = data_dir
        self.processor = processor
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        
        # 증강된 데이터 파일들 찾기
        self.episode_dirs = []
        for item in os.listdir(data_dir):
            item_path = os.path.join(data_dir, item)
            if os.path.isdir(item_path):
                self.episode_dirs.append(item_path)
        
        logger.info(f"📊 로드된 증강 에피소드 수: {len(self.episode_dirs)}")
        
        # 모든 데이터 로드
        self.all_data = []
        for episode_dir in self.episode_dirs:
            episode_name = os.path.basename(episode_dir)
            
            # 이미지 파일들 찾기
            image_files = []
            for file in os.listdir(episode_dir):
                if file.endswith('.jpg') or file.endswith('.png'):
                    image_files.append(os.path.join(episode_dir, file))
            
            image_files = sorted(image_files)
            
            # 액션 파일 로드
            actions_file = os.path.join(episode_dir, 'actions.npy')
            if os.path.exists(actions_file):
                actions = np.load(actions_file)
                
                # 각 이미지와 액션을 매칭
                for i, image_file in enumerate(image_files):
                    if i < len(actions):
                        # 이미지 로드
                        img = Image.open(image_file).convert('RGB')
                        
                        # 액션 (linear_x, linear_y만 사용)
                        action = actions[i][:2]  # 2D 액션만
                        
                        # 텍스트 명령
                        text_command = f"Navigate around obstacle in {episode_name}"
                        
                        self.all_data.append({
                            'image': img,
                            'action': action,
                            'text': text_command
                        })
        
        logger.info(f"📊 총 증강 샘플 수: {len(self.all_data)}")
    
    def __len__(self):
        return len(self.all_data)
    
    def __getitem__(self, idx):
        data = self.all_data[idx]
        
        # 이미지 전처리
        image = self.transform(data['image'])
        
        # 텍스트 전처리 (CLIP 기본 길이 사용)
        text_inputs = self.processor(
            text=data['text'],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77  # CLIP 기본 최대 길이
        )
        
        # 텐서에서 스칼라 추출
        for key in text_inputs:
            text_inputs[key] = text_inputs[key].squeeze(0)
        
        # 액션을 텐서로 변환
        action = torch.tensor(data['action'], dtype=torch.float32)
        
        return {
            'image': image,
            'text_inputs': text_inputs,
            'action': action
        }

class OriginalCLIPModel(nn.Module):
    """Original 72 Episodes CLIP 모델"""
    
    def __init__(self, processor, hidden_dim=512, dropout=0.2):
        super().__init__()
        self.processor = processor
        
        # CLIP 모델들
        self.clip_vision = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_text = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        
        # Feature Fusion
        self.fusion = nn.Sequential(
            nn.Linear(768 + 512, hidden_dim),  # CLIP Vision + Text
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Action Head
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)  # 2D action
        )
    
    def forward(self, images, text_inputs):
        # CLIP Vision
        vision_outputs = self.clip_vision(images)
        vision_features = vision_outputs.pooler_output  # [batch, 768]
        
        # CLIP Text
        text_outputs = self.clip_text(**text_inputs)
        text_features = text_outputs.pooler_output  # [batch, 512]
        
        # Feature Fusion
        combined = torch.cat([vision_features, text_features], dim=1)
        fused = self.fusion(combined)
        
        # Action Prediction
        actions = self.action_head(fused)
        
        return actions

class Trainer:
    """훈련기"""
    
    def __init__(self, model, device, learning_rate=5e-5, weight_decay=1e-3):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)
        self.criterion = nn.MSELoss()
    
    def train_step(self, batch):
        self.model.train()
        
        images = batch['image'].to(self.device)
        text_inputs = {k: v.to(self.device) for k, v in batch['text_inputs'].items()}
        targets = batch['action'].to(self.device)
        
        self.optimizer.zero_grad()
        
        predictions = self.model(images, text_inputs)
        loss = self.criterion(predictions, targets)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                images = batch['image'].to(self.device)
                text_inputs = {k: v.to(self.device) for k, v in batch['text_inputs'].items()}
                targets = batch['action'].to(self.device)
                
                predictions = self.model(images, text_inputs)
                loss = self.criterion(predictions, targets)
                
                # MAE 계산
                mae = torch.mean(torch.abs(predictions - targets))
                
                total_loss += loss.item()
                total_mae += mae.item()
                num_batches += 1
        
        return total_loss / num_batches, total_mae / num_batches

def train_original_clip_augmented(data_path, output_dir, num_epochs=10, batch_size=4, 
                                learning_rate=5e-5, weight_decay=1e-3):
    """Original CLIP 모델을 증강 데이터로 훈련"""
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🔧 디바이스: {device}")
    
    # 출력 디렉토리 생성
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 프로세서 로드
    logger.info("📥 CLIP 프로세서 로드 중...")
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # 데이터셋 생성
    logger.info("📊 증강 데이터셋 생성 중...")
    dataset = AugmentedDataset(data_path, processor)
    
    # 커스텀 collate 함수
    def custom_collate(batch):
        images = torch.stack([item['image'] for item in batch])
        actions = torch.stack([item['action'] for item in batch])
        
        # 텍스트 입력들을 동일한 크기로 패딩
        text_inputs = {}
        for key in batch[0]['text_inputs'].keys():
            max_len = max(item['text_inputs'][key].size(0) for item in batch)
            padded_tensors = []
            for item in batch:
                tensor = item['text_inputs'][key]
                if tensor.size(0) < max_len:
                    # 패딩
                    padding = torch.zeros(max_len - tensor.size(0), dtype=tensor.dtype)
                    tensor = torch.cat([tensor, padding])
                padded_tensors.append(tensor)
            text_inputs[key] = torch.stack(padded_tensors)
        
        return {
            'image': images,
            'text_inputs': text_inputs,
            'action': actions
        }
    
    # 데이터 로더 생성
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=custom_collate)
    
    # 모델 생성
    logger.info("🤖 Original CLIP 모델 생성 중...")
    model = OriginalCLIPModel(processor, hidden_dim=512, dropout=0.2)
    
    # 훈련기 생성
    trainer = Trainer(model, device, learning_rate, weight_decay)
    
    # 훈련 기록
    train_losses = []
    val_losses = []
    val_maes = []
    best_mae = float('inf')
    
    logger.info(f"🎯 훈련 설정:")
    logger.info(f"   - 모델: Original CLIP (증강 데이터)")
    logger.info(f"   - 에포크: {num_epochs}")
    logger.info(f"   - 배치 크기: {batch_size}")
    logger.info(f"   - 학습률: {learning_rate}")
    logger.info(f"   - Weight Decay: {weight_decay}")
    
    # 훈련 루프
    for epoch in range(num_epochs):
        logger.info(f"\n📈 Epoch {epoch+1}/{num_epochs}")
        
        # 훈련 단계
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        train_pbar = tqdm(dataloader, desc=f"Training Epoch {epoch+1}")
        for batch in train_pbar:
            loss = trainer.train_step(batch)
            train_loss += loss
            train_batches += 1
            train_pbar.set_postfix({'Loss': f'{loss:.6f}'})
        
        avg_train_loss = train_loss / train_batches
        train_losses.append(avg_train_loss)
        
        # 검증 단계
        val_loss, val_mae = trainer.validate(dataloader)
        val_losses.append(val_loss)
        val_maes.append(val_mae)
        
        # 스케줄러 업데이트
        trainer.scheduler.step()
        
        # 로그 출력
        logger.info(f"   📊 훈련 손실: {avg_train_loss:.6f}")
        logger.info(f"   📊 검증 손실: {val_loss:.6f}")
        logger.info(f"   📊 검증 MAE: {val_mae:.6f}")
        logger.info(f"   📊 학습률: {trainer.optimizer.param_groups[0]['lr']:.2e}")
        
        # 최고 성능 체크포인트 저장
        if val_mae < best_mae:
            best_mae = val_mae
            best_checkpoint_path = output_path / f"best_original_clip_augmented_epoch_{epoch+1}.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'epoch': epoch + 1,
                'val_loss': val_loss,
                'val_mae': val_mae,
                'train_loss': avg_train_loss
            }, best_checkpoint_path)
            logger.info(f"   🏆 새로운 최고 성능! MAE: {best_mae:.6f}")
    
    # 최종 모델 저장
    final_checkpoint_path = output_path / "final_original_clip_augmented.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'epoch': epoch + 1,
        'val_loss': val_loss,
        'val_mae': val_mae,
        'train_loss': avg_train_loss
    }, final_checkpoint_path)
    
    # 훈련 결과 저장
    training_results = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_maes': val_maes,
        'best_mae': best_mae,
        'final_epoch': epoch + 1
    }
    
    with open(output_path / 'training_results.json', 'w') as f:
        json.dump(training_results, f, indent=2)
    
    logger.info(f"✅ Original CLIP (증강 데이터) 훈련 완료!")
    logger.info(f"   최고 MAE: {best_mae:.6f}")
    logger.info(f"   최종 MAE: {val_mae:.6f}")
    
    return model, trainer

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='Train Original CLIP Model with Augmented Data')
    parser.add_argument('--data_path', type=str, default='legacy/augmented_dataset', help='Path to augmented dataset')
    parser.add_argument('--output_dir', type=str, default='original_clip_augmented_results', help='Output directory')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay')
    
    args = parser.parse_args()
    
    # 훈련 실행
    model, trainer = train_original_clip_augmented(
        data_path=args.data_path,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )

if __name__ == "__main__":
    main()
