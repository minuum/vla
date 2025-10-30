#!/usr/bin/env python3
"""
MAE 0.212 모델의 실제 성능 검증 스크립트
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from pathlib import Path
import h5py
from PIL import Image
import torchvision.transforms as transforms
from transformers import AutoProcessor, AutoModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Kosmos2CLIPHybridModel(nn.Module):
    """Kosmos2+CLIP Hybrid 모델"""
    
    def __init__(self, processor, hidden_dim=512, dropout=0.2):
        super().__init__()
        self.processor = processor
        
        # Kosmos2 통합 모델
        self.kosmos2_model = AutoModel.from_pretrained("microsoft/kosmos-2-patch14-224")
        
        # CLIP 모델들
        self.clip_vision = AutoModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_text = AutoModel.from_pretrained("openai/clip-vit-base-patch32")
        
        # Feature Fusion (Kosmos2 + CLIP)
        self.fusion = nn.Sequential(
            nn.Linear(2048 + 512, hidden_dim),  # Kosmos2 + CLIP Text
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
        # Kosmos2 통합 처리
        kosmos2_inputs = self.processor(
            images=images,
            text=text_inputs,
            return_tensors="pt"
        )
        kosmos2_outputs = self.kosmos2_model(**kosmos2_inputs)
        kosmos2_features = kosmos2_outputs.pooler_output  # [batch, 2048]
        
        # CLIP Text
        clip_text_outputs = self.clip_text(**text_inputs)
        clip_text_features = clip_text_outputs.pooler_output  # [batch, 512]
        
        # Feature Fusion
        combined = torch.cat([kosmos2_features, clip_text_features], dim=1)
        fused = self.fusion(combined)
        
        # Action Prediction
        actions = self.action_head(fused)
        
        return actions

class Original72EpisodesDataset:
    """원본 72 에피소드 데이터셋"""
    def __init__(self, data_dir, processor):
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # HDF5 파일들 로드
        self.all_data = []
        h5_files = list(self.data_dir.glob("*.h5"))
        logger.info(f"📊 로드된 HDF5 파일 수: {len(h5_files)}")
        
        for h5_file in h5_files:
            with h5py.File(h5_file, 'r') as f:
                # 이미지 데이터
                images = f['images'][:]  # [num_frames, height, width, channels]
                actions = f['actions'][:]  # [num_frames, action_dim]
                
                # 각 프레임을 개별 샘플로 처리
                for i in range(len(images)):
                    image = Image.fromarray(images[i])
                    action = actions[i][:2]  # 2D 액션만 사용
                    
                    self.all_data.append({
                        'image': image,
                        'action': action,
                        'text': "Navigate around obstacles to track the target cup"
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

def calculate_mae(model, dataset, device):
    """MAE 계산"""
    model.eval()
    total_mae = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for i in range(min(100, len(dataset))):  # 처음 100개 샘플만 테스트
            sample = dataset[i]
            
            image = sample['image'].unsqueeze(0).to(device)
            text_inputs = {k: v.unsqueeze(0).to(device) for k, v in sample['text_inputs'].items()}
            target_action = sample['action'].unsqueeze(0).to(device)
            
            # 예측
            predicted_action = model(image, text_inputs)
            
            # MAE 계산
            mae = torch.mean(torch.abs(predicted_action - target_action)).item()
            total_mae += mae
            num_samples += 1
            
            if i % 20 == 0:
                logger.info(f"샘플 {i}: 예측={predicted_action[0].cpu().numpy()}, 실제={target_action[0].cpu().numpy()}, MAE={mae:.4f}")
    
    avg_mae = total_mae / num_samples
    return avg_mae

def main():
    logger.info("🔍 MAE 0.212 모델 성능 검증 시작")
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🔧 디바이스: {device}")
    
    # 프로세서 로드
    logger.info("📥 Kosmos2 프로세서 로드 중...")
    processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
    
    # 모델 생성
    logger.info("🤖 모델 생성 중...")
    model = Kosmos2CLIPHybridModel(processor)
    
    # 체크포인트 로드
    logger.info("📂 체크포인트 로드 중...")
    checkpoint_path = "results/simple_clip_lstm_results_extended/best_simple_clip_lstm_model.pth"
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 모델 가중치 로드
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    logger.info(f"✅ 체크포인트 로드 완료:")
    logger.info(f"   - 에포크: {checkpoint['epoch']}")
    logger.info(f"   - 저장된 MAE: {checkpoint['val_mae']:.4f}")
    logger.info(f"   - 데이터 경로: {checkpoint['args'].data_path}")
    
    # 데이터셋 로드
    logger.info("📊 데이터셋 로드 중...")
    dataset = Original72EpisodesDataset(checkpoint['args'].data_path, processor)
    
    # MAE 계산
    logger.info("📈 MAE 계산 중...")
    actual_mae = calculate_mae(model, dataset, device)
    
    logger.info("🎯 결과:")
    logger.info(f"   - 저장된 MAE: {checkpoint['val_mae']:.4f}")
    logger.info(f"   - 실제 측정 MAE: {actual_mae:.4f}")
    logger.info(f"   - 차이: {abs(checkpoint['val_mae'] - actual_mae):.4f}")
    
    if abs(checkpoint['val_mae'] - actual_mae) < 0.01:
        logger.info("✅ MAE 값이 일치합니다!")
    else:
        logger.warning("⚠️ MAE 값에 차이가 있습니다!")

if __name__ == "__main__":
    main()
