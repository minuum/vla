#!/usr/bin/env python3
"""
Mobile Image Encoder - mobile_vla_data_collector.py의 720p 이미지 처리 특화
RoboVLMs의 이미지 인코딩 기술을 Mobile VLA에 적용
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class MobileImageEncoder(nn.Module):
    """
    Mobile VLA 특화 이미지 인코더
    - 입력: [B, T, 3, 224, 224] (mobile_vla_data_collector.py에서 720p→224p 리사이즈됨)
    - 출력: [B, T, hidden_size] 시간적 특징
    """
    
    def __init__(
        self,
        backbone: str = "efficientnet_v2_s",
        hidden_size: int = 768,
        num_lstm_layers: int = 2,
        dropout: float = 0.1,
        freeze_backbone: bool = False
    ):
        super().__init__()
        
        self.backbone_name = backbone
        self.hidden_size = hidden_size
        self.num_lstm_layers = num_lstm_layers
        
        # 백본 CNN (EfficientNet V2 - 모바일 최적화)
        if backbone == "efficientnet_v2_s":
            self.backbone = models.efficientnet_v2_s(pretrained=True)
            backbone_output_size = 1000
        elif backbone == "resnet50":
            self.backbone = models.resnet50(pretrained=True)
            backbone_output_size = 2048
        elif backbone == "mobilenet_v3_large":
            self.backbone = models.mobilenet_v3_large(pretrained=True)  
            backbone_output_size = 1000
        else:
            raise ValueError(f"지원하지 않는 백본: {backbone}")
        
        # 백본 가중치 고정 옵션
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info(f"🔒 {backbone} 백본 가중치 고정됨")
        
        # CNN 특징을 hidden_size로 매핑
        self.feature_projection = nn.Sequential(
            nn.Linear(backbone_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 시간적 특징 추출 (18프레임 시퀀스)
        self.temporal_encoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,  # 양방향이므로 절반
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_lstm_layers > 1 else 0
        )
        
        # 출력 정규화
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        logger.info(f"🖼️ Mobile Image Encoder 초기화 완료")
        logger.info(f"   백본: {backbone}, Hidden: {hidden_size}, LSTM Layers: {num_lstm_layers}")
    
    def forward(self, image_sequence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_sequence: [B, T, 3, 224, 224] - 배치 크기 B, 시퀀스 길이 T
            
        Returns:
            temporal_features: [B, T, hidden_size] - 시간적 이미지 특징
        """
        B, T, C, H, W = image_sequence.shape
        
        # 배치와 시간 차원을 합쳐서 CNN에 입력
        images_flat = image_sequence.view(B * T, C, H, W)  # [B*T, 3, 224, 224]
        
        # CNN으로 각 프레임 특징 추출
        with torch.cuda.amp.autocast(enabled=True):  # Mixed precision
            frame_features = self.backbone(images_flat)  # [B*T, backbone_output_size]
        
        # 특징 차원을 hidden_size로 매핑
        frame_features = self.feature_projection(frame_features)  # [B*T, hidden_size]
        
        # 시간 차원 복원
        frame_features = frame_features.view(B, T, self.hidden_size)  # [B, T, hidden_size]
        
        # LSTM으로 시간적 특징 추출
        temporal_features, (hidden, cell) = self.temporal_encoder(frame_features)  # [B, T, hidden_size]
        
        # 레이어 정규화
        temporal_features = self.layer_norm(temporal_features)
        
        return temporal_features
    
    def extract_spatial_features(self, single_image: torch.Tensor) -> torch.Tensor:
        """단일 이미지에서 공간적 특징만 추출 (실시간 추론용)"""
        # single_image: [B, 3, 224, 224] 또는 [3, 224, 224]
        if single_image.dim() == 3:
            single_image = single_image.unsqueeze(0)  # [1, 3, 224, 224]
        
        with torch.cuda.amp.autocast(enabled=True):
            spatial_features = self.backbone(single_image)  # [B, backbone_output_size]
        
        spatial_features = self.feature_projection(spatial_features)  # [B, hidden_size]
        
        return spatial_features
    
    def get_feature_maps(self, image_sequence: torch.Tensor) -> dict:
        """중간 특징 맵들을 반환 (디버깅/분석용)"""
        B, T, C, H, W = image_sequence.shape
        images_flat = image_sequence.view(B * T, C, H, W)
        
        features = {}
        
        # EfficientNet의 중간 특징들 추출
        if self.backbone_name == "efficientnet_v2_s":
            x = images_flat
            for i, layer in enumerate(self.backbone.features):
                x = layer(x)
                if i in [2, 4, 6]:  # 선택된 레이어의 특징 저장
                    features[f"stage_{i}"] = x.view(B, T, *x.shape[1:])
        
        return features


class MobileImageEncoderLite(nn.Module):
    """
    경량화된 Mobile Image Encoder (Jetson용)
    더 작은 모델이지만 핵심 기능 유지
    """
    
    def __init__(
        self,
        hidden_size: int = 512,
        num_lstm_layers: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        # 경량화된 CNN 백본 (MobileNet V3)
        self.backbone = models.mobilenet_v3_small(pretrained=True)
        backbone_output_size = 1000
        
        # 특징 매핑 (더 작은 hidden_size)
        self.feature_projection = nn.Sequential(
            nn.Linear(backbone_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 간단한 GRU (LSTM보다 파라미터 적음)
        self.temporal_encoder = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0
        )
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        logger.info(f"🚀 Mobile Image Encoder Lite 초기화 (Hidden: {hidden_size})")
    
    def extract_spatial_features(self, single_image: torch.Tensor) -> torch.Tensor:
        """단일 이미지에서 공간적 특징만 추출 (실시간 추론용)"""
        if single_image.dim() == 3:
            single_image = single_image.unsqueeze(0)  # [1, 3, 224, 224]
        
        # 경량화된 CNN
        spatial_features = self.backbone(single_image)
        spatial_features = self.feature_projection(spatial_features)
        
        return spatial_features
    
    def forward(self, image_sequence: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = image_sequence.shape
        
        images_flat = image_sequence.view(B * T, C, H, W)
        
        # 경량화된 CNN
        frame_features = self.backbone(images_flat)
        frame_features = self.feature_projection(frame_features)
        frame_features = frame_features.view(B, T, self.hidden_size)
        
        # GRU로 시간적 특징
        temporal_features, hidden = self.temporal_encoder(frame_features)
        temporal_features = self.layer_norm(temporal_features)
        
        return temporal_features


if __name__ == "__main__":
    # 테스트 코드
    print("🧪 Mobile Image Encoder 테스트")
    
    # 모델 초기화
    encoder = MobileImageEncoder(hidden_size=768)
    encoder_lite = MobileImageEncoderLite(hidden_size=512)
    
    # 테스트 데이터 (mobile_vla_data_collector.py 형식)
    batch_size, seq_len = 2, 18
    test_images = torch.randn(batch_size, seq_len, 3, 224, 224)
    
    print(f"📊 입력 이미지: {test_images.shape}")
    
    # 일반 인코더 테스트
    with torch.no_grad():
        features = encoder(test_images)
        print(f"🖼️ 인코더 출력: {features.shape}")
        
        features_lite = encoder_lite(test_images)
        print(f"🚀 Lite 인코더 출력: {features_lite.shape}")
    
    # 파라미터 수 계산
    total_params = sum(p.numel() for p in encoder.parameters())
    lite_params = sum(p.numel() for p in encoder_lite.parameters())
    
    print(f"📊 일반 인코더 파라미터: {total_params:,}개 ({total_params/1e6:.1f}M)")
    print(f"🚀 Lite 인코더 파라미터: {lite_params:,}개 ({lite_params/1e6:.1f}M)")
    print(f"💡 파라미터 감소율: {(1 - lite_params/total_params)*100:.1f}%")
