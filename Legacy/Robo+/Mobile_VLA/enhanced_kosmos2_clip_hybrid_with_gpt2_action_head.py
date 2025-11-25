#!/usr/bin/env python3
"""
🎯 Enhanced Kosmos2+CLIP Hybrid with GPT2 Action Head
VLM + GPT2 Action Head 구조로 모바일 로봇 VLA 구현
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModel, GPT2Model, GPT2Config
import logging
from typing import Optional, Tuple, Dict, Any

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GPT2ActionHead(nn.Module):
    """GPT2 기반 Action Head"""
    
    def __init__(
        self,
        hidden_dim: int = 768,
        action_dim: int = 2,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_length: int = 512
    ):
        super().__init__()
        
        # GPT2 설정
        config = GPT2Config(
            vocab_size=1,  # 더미 값
            n_positions=max_length,
            n_ctx=max_length,
            n_embd=hidden_dim,
            n_layer=num_layers,
            n_head=num_heads,
            resid_pdrop=dropout,
            embd_pdrop=dropout,
            attn_pdrop=dropout,
            use_cache=False
        )
        
        # GPT2 모델 (가중치 초기화만 사용)
        self.gpt2 = GPT2Model(config)
        
        # Action projection layer
        self.action_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # Positional encoding for action sequence
        self.positional_encoding = nn.Parameter(
            torch.randn(max_length, hidden_dim) * 0.02
        )
        
        logger.info(f"GPT2 Action Head initialized:")
        logger.info(f"  - Hidden dim: {hidden_dim}")
        logger.info(f"  - Action dim: {action_dim}")
        logger.info(f"  - Num layers: {num_layers}")
        logger.info(f"  - Num heads: {num_heads}")
        logger.info(f"  - Max length: {max_length}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, hidden_dim] - VLM features
        Returns:
            actions: [batch_size, action_dim] - predicted actions
        """
        batch_size, seq_len, hidden_dim = x.shape
        
        # Add positional encoding
        if seq_len <= self.positional_encoding.size(0):
            pos_enc = self.positional_encoding[:seq_len].unsqueeze(0)
            x = x + pos_enc
        
        # GPT2 forward pass
        gpt2_output = self.gpt2(
            inputs_embeds=x,
            attention_mask=torch.ones(batch_size, seq_len, device=x.device)
        )
        
        # Use last hidden state for action prediction
        last_hidden = gpt2_output.last_hidden_state[:, -1, :]  # [batch_size, hidden_dim]
        
        # Project to action space
        actions = self.action_projection(last_hidden)
        
        return actions

class EnhancedKosmos2CLIPHybridWithGPT2ActionHead(nn.Module):
    """
    Enhanced Kosmos2+CLIP Hybrid Model with GPT2 Action Head
    
    아키텍처:
    - Kosmos2 Vision Encoder
    - CLIP Vision Encoder  
    - Vision Resampler
    - GPT2 Action Head
    """
    
    def __init__(
        self,
        action_dim: int = 2,  # 2D 액션 (linear_x, linear_y)
        vision_resampler_tokens: int = 64,
        hidden_dim: int = 768,
        gpt2_layers: int = 6,
        gpt2_heads: int = 8,
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
        
        # GPT2 Action Head
        self.gpt2_action_head = GPT2ActionHead(
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            num_layers=gpt2_layers,
            num_heads=gpt2_heads,
            dropout=dropout
        )
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"Enhanced Kosmos2+CLIP Hybrid Model with GPT2 Action Head initialized:")
        logger.info(f"  - Action dim: {action_dim}")
        logger.info(f"  - Vision resampler tokens: {vision_resampler_tokens}")
        logger.info(f"  - Hidden dim: {hidden_dim}")
        logger.info(f"  - GPT2 layers: {gpt2_layers}")
        logger.info(f"  - GPT2 heads: {gpt2_heads}")
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
            actions: [batch_size, action_dim] - 예측된 액션
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
        
        # 4. GPT2 Action Head로 액션 예측
        actions = self.gpt2_action_head(fused_features)
        
        return actions
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "model_name": "Enhanced Kosmos2+CLIP Hybrid with GPT2 Action Head",
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "mobile_optimized": self.mobile_optimized,
            "action_head_type": "GPT2"
        }

def create_model(
    action_dim: int = 2,
    vision_resampler_tokens: int = 64,
    hidden_dim: int = 768,
    gpt2_layers: int = 6,
    gpt2_heads: int = 8,
    dropout: float = 0.1,
    mobile_optimized: bool = True
) -> EnhancedKosmos2CLIPHybridWithGPT2ActionHead:
    """모델 생성 함수"""
    return EnhancedKosmos2CLIPHybridWithGPT2ActionHead(
        action_dim=action_dim,
        vision_resampler_tokens=vision_resampler_tokens,
        hidden_dim=hidden_dim,
        gpt2_layers=gpt2_layers,
        gpt2_heads=gpt2_heads,
        dropout=dropout,
        mobile_optimized=mobile_optimized
    )

if __name__ == "__main__":
    # 모델 테스트
    logger.info("Testing Enhanced Kosmos2+CLIP Hybrid with GPT2 Action Head...")
    
    # 모델 생성
    model = create_model(
        action_dim=2,
        vision_resampler_tokens=64,
        hidden_dim=768,
        gpt2_layers=6,
        gpt2_heads=8,
        dropout=0.1,
        mobile_optimized=True
    )
    
    # 테스트 입력
    batch_size = 2
    test_images = torch.randn(batch_size, 3, 224, 224)
    
    # Forward pass
    with torch.no_grad():
        actions = model(test_images)
    
    logger.info(f"Model test successful!")
    logger.info(f"Input shape: {test_images.shape}")
    logger.info(f"Output shape: {actions.shape}")
    logger.info(f"Model info: {model.get_model_info()}")
