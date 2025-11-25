#!/usr/bin/env python3
"""
🎯 Enhanced Kosmos2+CLIP Hybrid with Discrete Action Head
VLM + Discrete Action Head 구조로 모바일 로봇 VLA 구현
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModel
import logging
from typing import Optional, Tuple, Dict, Any
import math

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DiscreteActionHead(nn.Module):
    """Discrete Action Head - 이산 액션 공간 사용"""
    
    def __init__(
        self,
        hidden_dim: int = 768,
        action_dim: int = 2,
        num_discrete_actions: int = 100,  # 각 액션 차원당 이산 값 개수
        embedding_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.action_dim = action_dim
        self.num_discrete_actions = num_discrete_actions
        self.embedding_dim = embedding_dim
        
        # Action embedding layers
        self.action_embeddings = nn.ModuleList([
            nn.Embedding(num_discrete_actions, embedding_dim)
            for _ in range(action_dim)
        ])
        
        # Feature projection
        self.feature_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, embedding_dim)
        )
        
        # Action classifiers for each dimension
        self.action_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim * 2, embedding_dim),
                nn.LayerNorm(embedding_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embedding_dim, num_discrete_actions)
            )
            for _ in range(action_dim)
        ])
        
        # Continuous action decoder (이산 → 연속 변환)
        self.action_decoder = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim // 2),
                nn.GELU(),
                nn.Linear(embedding_dim // 2, 1)  # 연속 값 1개
            )
            for _ in range(action_dim)
        ])
        
        logger.info(f"Discrete Action Head initialized:")
        logger.info(f"  - Hidden dim: {hidden_dim}")
        logger.info(f"  - Action dim: {action_dim}")
        logger.info(f"  - Num discrete actions: {num_discrete_actions}")
        logger.info(f"  - Embedding dim: {embedding_dim}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, hidden_dim] - VLM features
        Returns:
            actions: [batch_size, action_dim] - predicted continuous actions
        """
        batch_size, seq_len, hidden_dim = x.shape
        
        # Global average pooling
        x_pooled = x.mean(dim=1)  # [batch_size, hidden_dim]
        
        # Feature projection
        projected_features = self.feature_projection(x_pooled)  # [batch_size, embedding_dim]
        
        # Discrete action prediction for each dimension
        discrete_actions = []
        continuous_actions = []
        
        for i in range(self.action_dim):
            # Get action embedding
            action_emb = self.action_embeddings[i].weight  # [num_discrete_actions, embedding_dim]
            
            # Compute similarity between features and action embeddings
            similarity = torch.matmul(projected_features, action_emb.T)  # [batch_size, num_discrete_actions]
            
            # Get discrete action probabilities
            discrete_logits = self.action_classifiers[i](
                torch.cat([projected_features, action_emb.mean(dim=0).unsqueeze(0).expand(batch_size, -1)], dim=-1)
            )
            discrete_probs = F.softmax(discrete_logits, dim=-1)
            
            # Sample discrete action
            discrete_action = torch.multinomial(discrete_probs, 1).squeeze(-1)  # [batch_size]
            discrete_actions.append(discrete_action)
            
            # Convert discrete action to continuous
            discrete_emb = self.action_embeddings[i](discrete_action)  # [batch_size, embedding_dim]
            continuous_action = self.action_decoder[i](discrete_emb).squeeze(-1)  # [batch_size]
            continuous_actions.append(continuous_action)
        
        # Stack continuous actions
        continuous_actions = torch.stack(continuous_actions, dim=-1)  # [batch_size, action_dim]
        
        return continuous_actions
    
    def get_discrete_actions(self, x: torch.Tensor) -> torch.Tensor:
        """이산 액션만 반환 (디버깅용)"""
        batch_size, seq_len, hidden_dim = x.shape
        x_pooled = x.mean(dim=1)
        projected_features = self.feature_projection(x_pooled)
        
        discrete_actions = []
        for i in range(self.action_dim):
            discrete_logits = self.action_classifiers[i](
                torch.cat([projected_features, self.action_embeddings[i].weight.mean(dim=0).unsqueeze(0).expand(batch_size, -1)], dim=-1)
            )
            discrete_probs = F.softmax(discrete_logits, dim=-1)
            discrete_action = torch.multinomial(discrete_probs, 1).squeeze(-1)
            discrete_actions.append(discrete_action)
        
        return torch.stack(discrete_actions, dim=-1)

class EnhancedKosmos2CLIPHybridWithDiscreteActionHead(nn.Module):
    """
    Enhanced Kosmos2+CLIP Hybrid Model with Discrete Action Head
    
    아키텍처:
    - Kosmos2 Vision Encoder
    - CLIP Vision Encoder  
    - Vision Resampler
    - Discrete Action Head
    """
    
    def __init__(
        self,
        action_dim: int = 2,  # 2D 액션 (linear_x, linear_y)
        vision_resampler_tokens: int = 64,
        hidden_dim: int = 768,
        num_discrete_actions: int = 100,
        embedding_dim: int = 256,
        dropout: float = 0.1,
        mobile_optimized: bool = True
    ):
        super().__init__()
        
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.mobile_optimized = mobile_optimized
        
        # Kosmos2 모델 (Vision Encoder)
        logger.info("Loading Kosmos2 model...")
        self.kosmos_processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
        self.kosmos_model = AutoModel.from_pretrained("microsoft/kosmos-2-patch14-224")
        
        # CLIP 모델 (Vision Encoder)
        logger.info("Loading CLIP model...")
        self.clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model = AutoModel.from_pretrained("openai/clip-vit-base-patch32")
        
        # Vision Resampler (메모리 효율성을 위한 토큰 수 감소)
        self.vision_resampler = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Resampler query tokens
        self.resampler_queries = nn.Parameter(
            torch.randn(vision_resampler_tokens, hidden_dim) * 0.02
        )
        
        # Feature fusion layer
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Discrete Action Head
        self.discrete_action_head = DiscreteActionHead(
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            num_discrete_actions=num_discrete_actions,
            embedding_dim=embedding_dim,
            dropout=dropout
        )
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"Enhanced Kosmos2+CLIP Hybrid Model with Discrete Action Head initialized:")
        logger.info(f"  - Action dim: {action_dim}")
        logger.info(f"  - Vision resampler tokens: {vision_resampler_tokens}")
        logger.info(f"  - Hidden dim: {hidden_dim}")
        logger.info(f"  - Num discrete actions: {num_discrete_actions}")
        logger.info(f"  - Embedding dim: {embedding_dim}")
        logger.info(f"  - Mobile optimized: {mobile_optimized}")
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def extract_kosmos_features(self, images: torch.Tensor) -> torch.Tensor:
        """Kosmos2에서 vision features 추출"""
        with torch.no_grad():
            # Kosmos2 vision encoder 사용
            vision_outputs = self.kosmos_model.vision_model(images)
            # [batch_size, num_patches, hidden_dim]
            return vision_outputs.last_hidden_state
    
    def extract_clip_features(self, images: torch.Tensor) -> torch.Tensor:
        """CLIP에서 vision features 추출"""
        with torch.no_grad():
            # CLIP vision encoder 사용
            vision_outputs = self.clip_model.vision_model(images)
            # [batch_size, num_patches, hidden_dim]
            return vision_outputs.last_hidden_state
    
    def resample_vision_features(self, features: torch.Tensor) -> torch.Tensor:
        """Vision features를 고정된 토큰 수로 리샘플링"""
        batch_size = features.size(0)
        
        # Query tokens를 배치 크기만큼 복제
        queries = self.resampler_queries.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Multi-head attention으로 리샘플링
        resampled_features, _ = self.vision_resampler(
            query=queries,
            key=features,
            value=features
        )
        
        return resampled_features
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            images: [batch_size, channels, height, width] - 입력 이미지
        Returns:
            actions: [batch_size, action_dim] - 예측된 연속 액션
        """
        # 1. Vision features 추출
        kosmos_features = self.extract_kosmos_features(images)  # [B, N, 768]
        clip_features = self.extract_clip_features(images)      # [B, M, 768]
        
        # 2. Vision features 리샘플링 (메모리 효율성)
        kosmos_resampled = self.resample_vision_features(kosmos_features)
        clip_resampled = self.resample_vision_features(clip_features)
        
        # 3. Feature fusion
        fused_features = torch.cat([kosmos_resampled, clip_resampled], dim=-1)
        fused_features = self.feature_fusion(fused_features)
        
        # 4. Discrete Action Head로 액션 예측
        actions = self.discrete_action_head(fused_features)
        
        return actions
    
    def get_discrete_actions(self, images: torch.Tensor) -> torch.Tensor:
        """이산 액션 반환 (디버깅용)"""
        # 1-3. Vision features 추출 및 fusion (forward와 동일)
        kosmos_features = self.extract_kosmos_features(images)
        clip_features = self.extract_clip_features(images)
        kosmos_resampled = self.resample_vision_features(kosmos_features)
        clip_resampled = self.resample_vision_features(clip_features)
        fused_features = torch.cat([kosmos_resampled, clip_resampled], dim=-1)
        fused_features = self.feature_fusion(fused_features)
        
        # 4. Discrete actions만 반환
        return self.discrete_action_head.get_discrete_actions(fused_features)
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "model_name": "Enhanced Kosmos2+CLIP Hybrid with Discrete Action Head",
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "mobile_optimized": self.mobile_optimized,
            "action_head_type": "Discrete"
        }

def create_model(
    action_dim: int = 2,
    vision_resampler_tokens: int = 64,
    hidden_dim: int = 768,
    num_discrete_actions: int = 100,
    embedding_dim: int = 256,
    dropout: float = 0.1,
    mobile_optimized: bool = True
) -> EnhancedKosmos2CLIPHybridWithDiscreteActionHead:
    """모델 생성 함수"""
    return EnhancedKosmos2CLIPHybridWithDiscreteActionHead(
        action_dim=action_dim,
        vision_resampler_tokens=vision_resampler_tokens,
        hidden_dim=hidden_dim,
        num_discrete_actions=num_discrete_actions,
        embedding_dim=embedding_dim,
        dropout=dropout,
        mobile_optimized=mobile_optimized
    )

if __name__ == "__main__":
    # 모델 테스트
    logger.info("Testing Enhanced Kosmos2+CLIP Hybrid with Discrete Action Head...")
    
    # 모델 생성
    model = create_model(
        action_dim=2,
        vision_resampler_tokens=64,
        hidden_dim=768,
        num_discrete_actions=100,
        embedding_dim=256,
        dropout=0.1,
        mobile_optimized=True
    )
    
    # 테스트 입력
    batch_size = 2
    test_images = torch.randn(batch_size, 3, 224, 224)
    
    # Forward pass
    with torch.no_grad():
        continuous_actions = model(test_images)
        discrete_actions = model.get_discrete_actions(test_images)
    
    logger.info(f"Model test successful!")
    logger.info(f"Input shape: {test_images.shape}")
    logger.info(f"Continuous actions shape: {continuous_actions.shape}")
    logger.info(f"Discrete actions shape: {discrete_actions.shape}")
    logger.info(f"Model info: {model.get_model_info()}")
