#!/usr/bin/env python3
"""
원본 72개 데이터셋을 사용한 간단한 CLIP 기반 모델들 학습
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

class Original72EpisodesDataset(Dataset):
    """원본 72개 에피소드 데이터셋"""
    
    def __init__(self, data_dir, processor, max_episodes=72, transform=None):
        self.data_dir = data_dir
        self.processor = processor
        self.max_episodes = max_episodes
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        
        # .h5 파일들 찾기
        self.episode_files = []
        for file in os.listdir(data_dir):
            if file.endswith('.h5'):
                self.episode_files.append(os.path.join(data_dir, file))
        
        self.episode_files = sorted(self.episode_files)[:max_episodes]
        logger.info(f"📊 로드된 에피소드 수: {len(self.episode_files)}")
        
        # 모든 데이터 로드
        self.all_data = []
        for episode_file in self.episode_files:
            with h5py.File(episode_file, 'r') as f:
                images = f['images'][:]  # (18, 720, 1280, 3)
                actions = f['actions'][:]  # (18, 3)
                
                # 각 프레임을 개별 샘플로 처리
                for i in range(len(images)):
                    # 이미지를 PIL Image로 변환
                    img = Image.fromarray(images[i])
                    
                    # 액션 (linear_x, linear_y만 사용)
                    action = actions[i][:2]  # 2D 액션만
                    
                    # 텍스트 명령 (에피소드 파일명에서 추출)
                    episode_name = os.path.basename(episode_file).replace('.h5', '')
                    text_command = f"Navigate around obstacle in {episode_name}"
                    
                    self.all_data.append({
                        'image': img,
                        'action': action,
                        'text': text_command
                    })
        
        logger.info(f"📊 총 샘플 수: {len(self.all_data)}")
    
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

class SimpleCLIPModel(nn.Module):
    """간단한 CLIP 기반 모델 (기본)"""
    
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

class CLIPWithLSTMModel(nn.Module):
    """CLIP + LSTM 모델 (Kosmos2+CLIP Hybrid 대체)"""
    
    def __init__(self, processor, hidden_dim=512, dropout=0.2):
        super().__init__()
        self.processor = processor
        
        # CLIP 모델들
        self.clip_vision = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_text = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        
        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size=768 + 512,  # CLIP Vision + Text
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout
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
        
        # LSTM Processing (sequence length = 1)
        lstm_input = combined.unsqueeze(1)  # [batch, 1, features]
        lstm_out, _ = self.lstm(lstm_input)
        lstm_features = lstm_out.squeeze(1)  # [batch, hidden_dim]
        
        # Action Prediction
        actions = self.action_head(lstm_features)
        
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

def train_model(model_name, model_class, data_path, output_dir, num_epochs=10, batch_size=4, 
               learning_rate=5e-5, weight_decay=1e-3):
    """모델 훈련 함수"""
    
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
    logger.info("📊 데이터셋 생성 중...")
    dataset = Original72EpisodesDataset(data_path, processor, max_episodes=72)
    
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
    logger.info(f"🤖 {model_name} 모델 생성 중...")
    model = model_class(processor, hidden_dim=512, dropout=0.2)
    
    # 훈련기 생성
    trainer = Trainer(model, device, learning_rate, weight_decay)
    
    # 훈련 기록
    train_losses = []
    val_losses = []
    val_maes = []
    best_mae = float('inf')
    
    logger.info(f"🎯 훈련 설정:")
    logger.info(f"   - 모델: {model_name}")
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
            best_checkpoint_path = output_path / f"best_{model_name.lower().replace(' ', '_')}_epoch_{epoch+1}.pth"
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
    final_checkpoint_path = output_path / f"final_{model_name.lower().replace(' ', '_')}.pth"
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
        'model_name': model_name,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_maes': val_maes,
        'best_mae': best_mae,
        'final_epoch': epoch + 1
    }
    
    with open(output_path / 'training_results.json', 'w') as f:
        json.dump(training_results, f, indent=2)
    
    logger.info(f"✅ {model_name} 훈련 완료!")
    logger.info(f"   최고 MAE: {best_mae:.6f}")
    logger.info(f"   최종 MAE: {val_mae:.6f}")
    
    return model, trainer, best_mae

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='Train Simple CLIP Models on Original Dataset')
    parser.add_argument('--data_path', type=str, default='mobile_vla_dataset', help='Path to dataset')
    parser.add_argument('--output_dir', type=str, default='simple_models_original_results', help='Output directory')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay')
    
    args = parser.parse_args()
    
    # 모델 정의
    models = [
        ("Simple CLIP", SimpleCLIPModel),
        ("CLIP with LSTM", CLIPWithLSTMModel)
    ]
    
    results = {}
    
    # 각 모델 훈련
    for model_name, model_class in models:
        logger.info(f"\n{'='*60}")
        logger.info(f"🚀 {model_name} 모델 훈련 시작")
        logger.info(f"{'='*60}")
        
        output_dir = f"{args.output_dir}/{model_name.lower().replace(' ', '_')}"
        
        try:
            model, trainer, best_mae = train_model(
                model_name=model_name,
                model_class=model_class,
                data_path=args.data_path,
                output_dir=output_dir,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay
            )
            results[model_name] = best_mae
        except Exception as e:
            logger.error(f"❌ {model_name} 훈련 실패: {e}")
            results[model_name] = "실패"
    
    # 결과 요약
    logger.info(f"\n{'='*60}")
    logger.info("📊 최종 결과 요약")
    logger.info(f"{'='*60}")
    
    for model_name, mae in results.items():
        if isinstance(mae, float):
            logger.info(f"   {model_name}: MAE {mae:.6f}")
        else:
            logger.info(f"   {model_name}: {mae}")
    
    # 결과 저장
    with open(f"{args.output_dir}/final_results_summary.json", 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
