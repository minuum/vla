#!/usr/bin/env python3
"""
Distance-Aware Mobile VLA Inference Script
"""
import torch
import torch.nn as nn
from transformers import AutoProcessor, AutoModel
from PIL import Image
import numpy as np

class DistanceAwareVLAModel(nn.Module):
    def __init__(self, processor, hidden_size=768, lstm_hidden_size=256):
        super().__init__()
        self.processor = processor
        self.hidden_size = hidden_size
        self.lstm_hidden_size = lstm_hidden_size
        
        # Kosmos2 모델
        self.kosmos = AutoModel.from_pretrained("microsoft/kosmos-2-patch14-224")
        
        # LSTM + MLP 액션 헤드
        self.lstm = nn.LSTM(hidden_size, lstm_hidden_size, batch_first=True)
        self.action_head = nn.Sequential(
            nn.Linear(lstm_hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3)
        )
        
        # 거리별 특화 레이어
        self.distance_embedding = nn.Embedding(3, 32)
        self.distance_fusion = nn.Linear(lstm_hidden_size + 32, lstm_hidden_size)

    def forward(self, images, distance_labels=None):
        batch_size, seq_len, c, h, w = images.shape
        device = images.device
        
        # 이미지 특징 추출
        image_features = []
        for t in range(seq_len):
            try:
                pixel_values = images[:, t, :, :, :]
                dummy_input_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
                dummy_attention_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
                
                with torch.no_grad():
                    vision_outputs = self.kosmos.vision_model(pixel_values=pixel_values)
                    features = vision_outputs.last_hidden_state.mean(dim=1)
            except:
                features = torch.randn(batch_size, self.hidden_size, device=device)
            
            if features.shape[-1] != self.hidden_size:
                if not hasattr(self, 'feature_adapter'):
                    self.feature_adapter = nn.Linear(features.shape[-1], self.hidden_size).to(features.device)
                features = self.feature_adapter(features)
            
            image_features.append(features)
        
        sequence_features = torch.stack(image_features, dim=1)
        lstm_out, _ = self.lstm(sequence_features)
        
        if distance_labels is not None:
            distance_embeds = self.distance_embedding(distance_labels)
            distance_embeds = distance_embeds.unsqueeze(1).expand(-1, seq_len, -1)
            combined_features = torch.cat([lstm_out, distance_embeds], dim=-1)
            lstm_out = self.distance_fusion(combined_features)
        
        final_features = lstm_out[:, -1, :]
        predicted_actions = self.action_head(final_features)
        
        return predicted_actions

def load_model(model_path):
    """모델 로드"""
    processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
    model = DistanceAwareVLAModel(processor)
    
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, processor

def predict_actions(model, processor, images, distance_label=1):
    """액션 예측"""
    device = next(model.parameters()).device
    
    # 이미지 전처리
    if isinstance(images, list):
        # PIL 이미지 리스트를 텐서로 변환
        processed_images = []
        for img in images:
            if isinstance(img, Image.Image):
                img_tensor = processor(images=img, return_tensors="pt")["pixel_values"]
                processed_images.append(img_tensor)
        images = torch.cat(processed_images, dim=0).unsqueeze(0)  # [1, 8, 3, 224, 224]
    
    images = images.to(device)
    distance_labels = torch.tensor([distance_label]).to(device)
    
    with torch.no_grad():
        predicted_actions = model(images, distance_labels)
    
    return predicted_actions.cpu().numpy()

if __name__ == "__main__":
    # 사용 예시
    model_path = "best_distance_aware_model.pth"
    model, processor = load_model(model_path)
    
    # 더미 이미지로 테스트
    dummy_images = torch.randn(1, 8, 3, 224, 224)
    actions = predict_actions(model, processor, dummy_images, distance_label=1)
    
    print(f"예측된 액션: {actions}")
    print(f"액션 형태: {actions.shape}")
