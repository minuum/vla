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
        self.kosmos2_model = AutoModel.from_pretrained("microsoft/kosmos-2-patch14-224")
        self.clip_vision = AutoModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_text = AutoModel.from_pretrained("openai/clip-vit-base-patch32")
        
        # 융합 레이어
        kosmos_dim = self.kosmos2_model.config.hidden_size
        clip_dim = self.clip_vision.config.hidden_size
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(kosmos_dim + clip_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # LSTM 레이어
        self.lstm = nn.LSTM(
            input_size=hidden_dim // 2,
            hidden_size=hidden_dim,
            num_layers=4,
            batch_first=True,
            dropout=dropout
        )
        
        # 액션 예측 헤드
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)  # 2D 액션
        )
        
    def forward(self, images, texts):
        batch_size = images.size(0)
        
        # Kosmos2 처리
        kosmos_inputs = self.processor(
            images=images,
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        # 디바이스로 이동
        for key in kosmos_inputs:
            if isinstance(kosmos_inputs[key], torch.Tensor):
                kosmos_inputs[key] = kosmos_inputs[key].to(images.device)
        
        kosmos_outputs = self.kosmos2_model(**kosmos_inputs)
        kosmos_features = kosmos_outputs.last_hidden_state.mean(dim=1)  # [batch_size, kosmos_dim]
        
        # CLIP Vision 처리
        clip_vision_outputs = self.clip_vision(pixel_values=images)
        clip_vision_features = clip_vision_outputs.pooler_output  # [batch_size, clip_dim]
        
        # CLIP Text 처리
        clip_text_inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        for key in clip_text_inputs:
            if isinstance(clip_text_inputs[key], torch.Tensor):
                clip_text_inputs[key] = clip_text_inputs[key].to(images.device)
        
        clip_text_outputs = self.clip_text(**clip_text_inputs)
        clip_text_features = clip_text_outputs.pooler_output  # [batch_size, clip_dim]
        
        # 특징 융합
        combined_features = torch.cat([
            kosmos_features,
            clip_vision_features,
            clip_text_features
        ], dim=1)
        
        fused_features = self.fusion_layer(combined_features)
        
        # LSTM 처리 (시퀀스로 변환)
        sequence_features = fused_features.unsqueeze(1)  # [batch_size, 1, hidden_dim//2]
        lstm_out, _ = self.lstm(sequence_features)
        
        # 액션 예측
        actions = self.action_head(lstm_out.squeeze(1))
        
        return actions

class Original72EpisodesDataset:
    """원본 72 에피소드 데이터셋"""
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
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
        
        # 액션을 텐서로 변환
        action = torch.tensor(data['action'], dtype=torch.float32)
        
        return {
            'image': image,
            'text': data['text'],
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
            text = [sample['text']]
            target_action = sample['action'].unsqueeze(0).to(device)
            
            # 예측
            predicted_action = model(image, text)
            
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
    dataset = Original72EpisodesDataset(checkpoint['args'].data_path)
    
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
