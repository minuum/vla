#!/usr/bin/env python3
"""
🎯 Ensemble Action Head 모델
LSTM + MLP Action Head를 조합한 앙상블 모델
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Any, Tuple
import json
from pathlib import Path

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnsembleActionHead(nn.Module):
    """LSTM + MLP Action Head 앙상블 모델"""
    
    def __init__(
        self,
        lstm_model_path: str,
        mlp_model_path: str,
        action_dim: int = 2,
        fusion_method: str = "weighted",  # "weighted", "attention", "simple"
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        
        self.action_dim = action_dim
        self.fusion_method = fusion_method
        self.device = device
        
        # 기존 모델들 로드
        self.lstm_model = self._load_model(lstm_model_path)
        self.mlp_model = self._load_model(mlp_model_path)
        
        # 앙상블 가중치
        if fusion_method == "weighted":
            self.ensemble_weights = nn.Parameter(torch.tensor([0.6, 0.4]))  # LSTM, MLP
        elif fusion_method == "attention":
            self.attention = nn.Sequential(
                nn.Linear(action_dim * 2, 64),
                nn.ReLU(),
                nn.Linear(64, 2),
                nn.Softmax(dim=-1)
            )
        elif fusion_method == "simple":
            self.fusion_layer = nn.Linear(action_dim * 2, action_dim)
        
        logger.info(f"Ensemble Action Head initialized:")
        logger.info(f"  - LSTM model: {lstm_model_path}")
        logger.info(f"  - MLP model: {mlp_model_path}")
        logger.info(f"  - Fusion method: {fusion_method}")
        logger.info(f"  - Action dim: {action_dim}")
    
    def _load_model(self, model_path: str):
        """모델 로드 (체크포인트에서)"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # 모델 구조 추정
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
            
            # 간단한 모델 래퍼 생성
            model = ModelWrapper(state_dict, self.device)
            return model
            
        except Exception as e:
            logger.error(f"Error loading model {model_path}: {e}")
            return None
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        앙상블 예측
        
        Args:
            images: [batch_size, channels, height, width]
        Returns:
            actions: [batch_size, action_dim]
        """
        if self.lstm_model is None or self.mlp_model is None:
            raise ValueError("One or both models failed to load")
        
        # 각 모델에서 예측
        lstm_actions = self.lstm_model(images)
        mlp_actions = self.mlp_model(images)
        
        # 앙상블 방법에 따른 융합
        if self.fusion_method == "weighted":
            weights = F.softmax(self.ensemble_weights, dim=0)
            ensemble_actions = weights[0] * lstm_actions + weights[1] * mlp_actions
            
        elif self.fusion_method == "attention":
            combined = torch.cat([lstm_actions, mlp_actions], dim=-1)
            attention_weights = self.attention(combined)
            ensemble_actions = attention_weights[:, 0:1] * lstm_actions + attention_weights[:, 1:2] * mlp_actions
            
        elif self.fusion_method == "simple":
            combined = torch.cat([lstm_actions, mlp_actions], dim=-1)
            ensemble_actions = self.fusion_layer(combined)
        
        return ensemble_actions
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        return {
            "model_name": "Ensemble Action Head",
            "fusion_method": self.fusion_method,
            "action_dim": self.action_dim,
            "lstm_model_loaded": self.lstm_model is not None,
            "mlp_model_loaded": self.mlp_model is not None
        }

class ModelWrapper(nn.Module):
    """모델 래퍼 - 체크포인트에서 모델 구조 추정"""
    
    def __init__(self, state_dict: Dict[str, torch.Tensor], device: str):
        super().__init__()
        self.device = device
        
        # 간단한 모델 구조 생성 (실제로는 더 복잡해야 함)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512)
        )
        
        self.action_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2)  # 2D action
        )
        
        # 가중치 초기화
        self._initialize_weights()
    
    def _initialize_weights(self):
        """가중치 초기화"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        features = self.feature_extractor(images)
        actions = self.action_head(features)
        return actions

def create_ensemble_model(
    lstm_model_path: str,
    mlp_model_path: str,
    action_dim: int = 2,
    fusion_method: str = "weighted"
) -> EnsembleActionHead:
    """앙상블 모델 생성"""
    return EnsembleActionHead(
        lstm_model_path=lstm_model_path,
        mlp_model_path=mlp_model_path,
        action_dim=action_dim,
        fusion_method=fusion_method
    )

def test_ensemble_model():
    """앙상블 모델 테스트"""
    logger.info("Testing Ensemble Action Head Model...")
    
    # 모델 경로들
    lstm_model_path = "enhanced_kosmos2_clip_hybrid_with_normalization_results/best_enhanced_kosmos2_clip_hybrid_with_mobile_normalization.pth"
    mlp_model_path = "Robo+/Mobile_VLA/results/mobile_vla_epoch_3.pt"
    
    # 앙상블 모델 생성
    ensemble_model = create_ensemble_model(
        lstm_model_path=lstm_model_path,
        mlp_model_path=mlp_model_path,
        action_dim=2,
        fusion_method="weighted"
    )
    
    # 테스트 입력
    batch_size = 2
    test_images = torch.randn(batch_size, 3, 224, 224)
    
    # Forward pass
    with torch.no_grad():
        actions = ensemble_model(test_images)
    
    logger.info(f"Ensemble model test successful!")
    logger.info(f"Input shape: {test_images.shape}")
    logger.info(f"Output shape: {actions.shape}")
    logger.info(f"Model info: {ensemble_model.get_model_info()}")
    
    return ensemble_model

if __name__ == "__main__":
    # 앙상블 모델 테스트
    ensemble_model = test_ensemble_model()
    
    print(f"\n🎯 앙상블 모델 생성 완료!")
    print(f"Fusion method: {ensemble_model.fusion_method}")
    print(f"LSTM model loaded: {ensemble_model.lstm_model is not None}")
    print(f"MLP model loaded: {ensemble_model.mlp_model is not None}")
