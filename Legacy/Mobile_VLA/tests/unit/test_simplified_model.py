#!/usr/bin/env python3
"""
Simplified 모델 테스트 스크립트
"""

import torch
import torch.nn as nn
from transformers import AutoProcessor, CLIPVisionModel, CLIPTextModel
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class SimpleTestModel(nn.Module):
    """간단한 테스트 모델"""
    
    def __init__(self):
        super().__init__()
        
        # CLIP 모델들
        self.clip_vision = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_text = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        
        # Feature Fusion
        self.fusion = nn.Sequential(
            nn.Linear(768 + 512, 512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Action Head
        self.action_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2)
        )
    
    def forward(self, images, text_inputs):
        # CLIP Vision
        vision_outputs = self.clip_vision(images)
        vision_features = vision_outputs.pooler_output
        
        # CLIP Text
        text_outputs = self.clip_text(**text_inputs)
        text_features = text_outputs.pooler_output
        
        # Feature Fusion
        combined = torch.cat([vision_features, text_features], dim=1)
        fused = self.fusion(combined)
        
        # Action Prediction
        actions = self.action_head(fused)
        
        return actions

def test_model():
    """모델 테스트"""
    print("🧪 Simplified 모델 테스트 시작")
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 디바이스: {device}")
    
    # 프로세서 로드
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # 모델 생성
    model = SimpleTestModel().to(device)
    model.eval()
    
    # 테스트 데이터 생성
    batch_size = 2
    
    # 더미 이미지 생성
    images = torch.randn(batch_size, 3, 224, 224).to(device)
    
    # 더미 텍스트 생성
    text_inputs = processor(
        text=["move forward", "turn left"],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)
    
    # 더미 액션 생성
    actions = torch.randn(batch_size, 2).to(device)
    
    print("📊 테스트 데이터 생성 완료")
    print(f"   이미지 크기: {images.shape}")
    print(f"   텍스트 입력 크기: {text_inputs['input_ids'].shape}")
    print(f"   액션 크기: {actions.shape}")
    
    # Forward pass 테스트
    try:
        with torch.no_grad():
            predicted_actions = model(images, text_inputs)
            print("✅ Forward pass 성공!")
            print(f"   예측 액션 크기: {predicted_actions.shape}")
            
            # 손실 계산 테스트
            criterion = nn.MSELoss()
            loss = criterion(predicted_actions, actions)
            print(f"   테스트 손실: {loss.item():.6f}")
            
    except Exception as e:
        print(f"❌ Forward pass 실패: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_model()
    if success:
        print("🎉 모델 테스트 성공!")
    else:
        print("💥 모델 테스트 실패!")
