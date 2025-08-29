#!/usr/bin/env python3
"""
Simplified RoboVLMs 모델 구조 분석 스크립트
"""

import torch
import torch.nn as nn
from transformers import AutoProcessor, CLIPVisionModel, CLIPTextModel
import json

def analyze_simplified_model():
    """Simplified RoboVLMs 모델 구조 분석"""
    
    checkpoint_path = "models/experimental/simplified_robovlms_best.pth"
    
    print("🔍 Simplified RoboVLMs 모델 구조 분석")
    print("="*60)
    
    # 체크포인트 로드
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('model_state_dict', {})
    
    print(f"✅ 체크포인트 로드 완료")
    print(f"   MAE: {checkpoint.get('val_mae', 'N/A')}")
    print(f"   에포크: {checkpoint.get('epoch', 'N/A')}")
    print(f"   모델 키 수: {len(state_dict)}")
    print()
    
    # 모델 구조 분석
    print("📊 모델 구조 분석")
    print("-"*40)
    
    # CLIP 관련 키들
    clip_vision_keys = [key for key in state_dict.keys() if 'clip_vision' in key.lower()]
    clip_text_keys = [key for key in state_dict.keys() if 'clip_text' in key.lower()]
    
    # 기타 컴포넌트들
    fusion_keys = [key for key in state_dict.keys() if 'fusion' in key.lower()]
    action_keys = [key for key in state_dict.keys() if 'action' in key.lower()]
    lstm_keys = [key for key in state_dict.keys() if 'lstm' in key.lower()]
    rnn_keys = [key for key in state_dict.keys() if 'rnn' in key.lower()]
    
    print(f"CLIP Vision 키: {len(clip_vision_keys)}개")
    print(f"CLIP Text 키: {len(clip_text_keys)}개")
    print(f"Fusion 키: {len(fusion_keys)}개")
    print(f"Action 키: {len(action_keys)}개")
    print(f"LSTM 키: {len(lstm_keys)}개")
    print(f"RNN 키: {len(rnn_keys)}개")
    print()
    
    # 주요 컴포넌트 상세 분석
    print("🔧 주요 컴포넌트 상세 분석")
    print("-"*40)
    
    # CLIP Vision 분석
    if clip_vision_keys:
        print("👁️ CLIP Vision 모델:")
        for key in clip_vision_keys[:5]:  # 처음 5개만
            shape = state_dict[key].shape
            print(f"   {key}: {shape}")
        if len(clip_vision_keys) > 5:
            print(f"   ... (총 {len(clip_vision_keys)}개)")
        print()
    
    # CLIP Text 분석
    if clip_text_keys:
        print("💬 CLIP Text 모델:")
        for key in clip_text_keys[:5]:  # 처음 5개만
            shape = state_dict[key].shape
            print(f"   {key}: {shape}")
        if len(clip_text_keys) > 5:
            print(f"   ... (총 {len(clip_text_keys)}개)")
        print()
    
    # Action Head 분석
    if action_keys:
        print("🎯 Action Head:")
        for key in action_keys:
            shape = state_dict[key].shape
            print(f"   {key}: {shape}")
        print()
    
    # RNN/LSTM 분석
    if rnn_keys or lstm_keys:
        print("🔄 RNN/LSTM:")
        for key in rnn_keys + lstm_keys:
            shape = state_dict[key].shape
            print(f"   {key}: {shape}")
        print()
    
    # 모델 구조 추정
    print("🏗️ 추정된 모델 구조")
    print("-"*40)
    
    # CLIP Vision 차원 추정
    if clip_vision_keys:
        for key in clip_vision_keys:
            if 'embeddings' in key and 'patch_embedding' in key:
                vision_dim = state_dict[key].shape[-1]
                print(f"CLIP Vision 차원: {vision_dim}")
                break
    
    # CLIP Text 차원 추정
    if clip_text_keys:
        for key in clip_text_keys:
            if 'embeddings' in key and 'token_embedding' in key:
                text_dim = state_dict[key].shape[-1]
                print(f"CLIP Text 차원: {text_dim}")
                break
    
    # Action 차원 추정
    if action_keys:
        for key in action_keys:
            if 'weight' in key and 'action' in key:
                if len(state_dict[key].shape) == 2:
                    action_dim = state_dict[key].shape[0]
                    print(f"Action 차원: {action_dim}")
                    break
    
    # RNN/LSTM 차원 추정
    if rnn_keys or lstm_keys:
        for key in rnn_keys + lstm_keys:
            if 'weight_ih_l0' in key:
                hidden_dim = state_dict[key].shape[0] // 4  # LSTM의 경우 4개 게이트
                print(f"RNN/LSTM Hidden 차원: {hidden_dim}")
                break
    
    print()
    
    # 모델 구조 재구성
    print("🔄 Simplified 모델 구조 재구성")
    print("-"*40)
    
    class SimplifiedRoboVLMsModel(nn.Module):
        def __init__(self, processor):
            super().__init__()
            self.processor = processor
            
            # CLIP 모델들
            self.clip_vision = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_text = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
            
            # Fusion 레이어
            self.fusion = nn.Sequential(
                nn.Linear(768 + 512, 512),  # CLIP Vision + Text
                nn.ReLU(),
                nn.Dropout(0.2)
            )
            
            # Action Head
            self.action_head = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 2)  # 2D action
            )
        
        def forward(self, images, text):
            # CLIP Vision
            vision_outputs = self.clip_vision(images)
            vision_features = vision_outputs.pooler_output  # [batch, 768]
            
            # CLIP Text
            text_outputs = self.clip_text(text)
            text_features = text_outputs.pooler_output  # [batch, 512]
            
            # Feature Fusion
            combined = torch.cat([vision_features, text_features], dim=1)
            fused = self.fusion(combined)
            
            # Action Prediction
            actions = self.action_head(fused)
            
            return actions
    
    print("✅ Simplified 모델 구조 재구성 완료")
    print("   - CLIP Vision + Text 기반")
    print("   - Feature Fusion")
    print("   - Simple Action Head")
    print("   - 2D Action 출력")
    
    return checkpoint

if __name__ == "__main__":
    analyze_simplified_model()
