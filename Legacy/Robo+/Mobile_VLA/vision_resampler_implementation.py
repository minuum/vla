#!/usr/bin/env python3
"""
Vision Resampler Implementation for Mobile VLA
RoboVLMs의 Vision Resampler를 모바일 환경에 최적화하여 구현

주요 기능:
1. 메모리 효율적인 Vision Token 처리
2. 동적 토큰 수 조절
3. 모바일 환경 최적화
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VisionResampler(nn.Module):
    """
    Vision Resampler for Mobile VLA
    
    RoboVLMs의 Vision Resampler를 모바일 환경에 최적화
    - 메모리 효율적인 토큰 처리
    - 동적 토큰 수 조절
    - 모바일 GPU 최적화
    """
    
    def __init__(
        self,
        input_dim: int = 768,  # CLIP ViT-B/32 output dimension
        output_dim: int = 768,  # Target dimension
        num_tokens: int = 64,   # Target number of tokens
        num_heads: int = 12,    # Number of attention heads
        dropout: float = 0.1,
        mobile_optimized: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_tokens = num_tokens
        self.num_heads = num_heads
        self.mobile_optimized = mobile_optimized
        
        # Learnable query tokens
        self.query_tokens = nn.Parameter(torch.randn(num_tokens, output_dim))
        nn.init.normal_(self.query_tokens, std=0.02)
        
        # Cross-attention layers
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(output_dim)
        self.norm2 = nn.LayerNorm(output_dim)
        
        # Feed-forward network
        if mobile_optimized:
            # 모바일 최적화: 더 작은 FFN
            self.ffn = nn.Sequential(
                nn.Linear(output_dim, output_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(output_dim * 2, output_dim)
            )
        else:
            # 표준 FFN
            self.ffn = nn.Sequential(
                nn.Linear(output_dim, output_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(output_dim * 4, output_dim)
            )
        
        # Input projection (if dimensions don't match)
        if input_dim != output_dim:
            self.input_proj = nn.Linear(input_dim, output_dim)
        else:
            self.input_proj = nn.Identity()
        
        # Output projection
        self.output_proj = nn.Linear(output_dim, output_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        logger.info(f"Vision Resampler initialized:")
        logger.info(f"  - Input dim: {input_dim}")
        logger.info(f"  - Output dim: {output_dim}")
        logger.info(f"  - Num tokens: {num_tokens}")
        logger.info(f"  - Mobile optimized: {mobile_optimized}")
    
    def forward(
        self, 
        vision_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of Vision Resampler
        
        Args:
            vision_features: [batch_size, seq_len, input_dim] Vision features
            attention_mask: Optional attention mask
            
        Returns:
            resampled_features: [batch_size, num_tokens, output_dim] Resampled features
        """
        batch_size, seq_len, _ = vision_features.shape
        
        # Input projection
        vision_features = self.input_proj(vision_features)
        
        # Expand query tokens for batch
        query_tokens = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Cross-attention: query_tokens attend to vision_features
        attn_output, attn_weights = self.cross_attention(
            query=query_tokens,
            key=vision_features,
            value=vision_features,
            key_padding_mask=attention_mask
        )
        
        # Residual connection and layer norm
        query_tokens = self.norm1(query_tokens + self.dropout(attn_output))
        
        # Feed-forward network
        ffn_output = self.ffn(query_tokens)
        
        # Residual connection and layer norm
        output = self.norm2(query_tokens + self.dropout(ffn_output))
        
        # Output projection
        output = self.output_proj(output)
        
        return output
    
    def get_attention_weights(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Get attention weights for visualization
        
        Args:
            vision_features: [batch_size, seq_len, input_dim] Vision features
            
        Returns:
            attention_weights: [batch_size, num_heads, num_tokens, seq_len] Attention weights
        """
        batch_size, seq_len, _ = vision_features.shape
        
        # Input projection
        vision_features = self.input_proj(vision_features)
        
        # Expand query tokens for batch
        query_tokens = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Get attention weights
        _, attn_weights = self.cross_attention(
            query=query_tokens,
            key=vision_features,
            value=vision_features,
            average_attn_weights=False
        )
        
        return attn_weights


class AdaptiveVisionResampler(VisionResampler):
    """
    Adaptive Vision Resampler with dynamic token count
    
    모바일 환경에서 동적으로 토큰 수를 조절하여
    성능과 효율성의 균형을 맞춤
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        output_dim: int = 768,
        min_tokens: int = 32,
        max_tokens: int = 128,
        num_heads: int = 12,
        dropout: float = 0.1,
        mobile_optimized: bool = True
    ):
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        
        # Initialize with max tokens
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            num_tokens=max_tokens,
            num_heads=num_heads,
            dropout=dropout,
            mobile_optimized=mobile_optimized
        )
        
        # Token selection network
        self.token_selector = nn.Sequential(
            nn.Linear(output_dim, output_dim // 2),
            nn.ReLU(),
            nn.Linear(output_dim // 2, 1),
            nn.Sigmoid()
        )
        
        logger.info(f"Adaptive Vision Resampler initialized:")
        logger.info(f"  - Min tokens: {min_tokens}")
        logger.info(f"  - Max tokens: {max_tokens}")
    
    def forward(
        self, 
        vision_features: torch.Tensor,
        target_tokens: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with adaptive token selection
        
        Args:
            vision_features: [batch_size, seq_len, input_dim] Vision features
            target_tokens: Target number of tokens (if None, auto-select)
            attention_mask: Optional attention mask
            
        Returns:
            resampled_features: [batch_size, selected_tokens, output_dim] Resampled features
        """
        # Get full resampled features
        full_features = super().forward(vision_features, attention_mask)
        
        if target_tokens is None:
            # Auto-select token count based on content
            target_tokens = self._auto_select_tokens(full_features)
        
        # Ensure target_tokens is within bounds
        target_tokens = max(self.min_tokens, min(target_tokens, self.max_tokens))
        
        if target_tokens == self.max_tokens:
            return full_features
        
        # Select top-k tokens
        selected_features = self._select_tokens(full_features, target_tokens)
        
        return selected_features
    
    def _auto_select_tokens(self, features: torch.Tensor) -> int:
        """
        Automatically select optimal number of tokens
        
        Args:
            features: [batch_size, max_tokens, output_dim] Full features
            
        Returns:
            optimal_tokens: Optimal number of tokens
        """
        # Calculate token importance scores
        importance_scores = self.token_selector(features)  # [batch_size, max_tokens, 1]
        importance_scores = importance_scores.squeeze(-1)  # [batch_size, max_tokens]
        
        # Average across batch
        avg_importance = importance_scores.mean(dim=0)  # [max_tokens]
        
        # Select tokens with importance > threshold
        threshold = 0.5
        selected_count = (avg_importance > threshold).sum().item()
        
        # Ensure within bounds
        selected_count = max(self.min_tokens, min(selected_count, self.max_tokens))
        
        return selected_count
    
    def _select_tokens(self, features: torch.Tensor, target_tokens: int) -> torch.Tensor:
        """
        Select top-k most important tokens
        
        Args:
            features: [batch_size, max_tokens, output_dim] Full features
            target_tokens: Number of tokens to select
            
        Returns:
            selected_features: [batch_size, target_tokens, output_dim] Selected features
        """
        # Calculate token importance scores
        importance_scores = self.token_selector(features)  # [batch_size, max_tokens, 1]
        importance_scores = importance_scores.squeeze(-1)  # [batch_size, max_tokens]
        
        # Get top-k indices
        _, top_indices = torch.topk(importance_scores, target_tokens, dim=1)
        
        # Select features
        batch_size = features.size(0)
        selected_features = torch.gather(
            features, 
            dim=1, 
            index=top_indices.unsqueeze(-1).expand(-1, -1, features.size(-1))
        )
        
        return selected_features


class MobileOptimizedVisionResampler(AdaptiveVisionResampler):
    """
    Mobile-optimized Vision Resampler
    
    Jetson Orin NX와 같은 모바일 환경에 특화된 최적화
    - 메모리 사용량 최소화
    - 연산 효율성 극대화
    - 실시간 처리 최적화
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        output_dim: int = 768,
        num_tokens: int = 64,
        num_heads: int = 8,  # Reduced for mobile
        dropout: float = 0.1
    ):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            min_tokens=32,
            max_tokens=num_tokens,
            num_heads=num_heads,
            dropout=dropout,
            mobile_optimized=True
        )
        
        # Mobile-specific optimizations
        self.use_half_precision = False  # Disable to avoid dtype mismatch
        self.use_gradient_checkpointing = True
        
        # Convert to half precision if supported
        if self.use_half_precision and torch.cuda.is_available():
            self.half()
        
        logger.info(f"Mobile-optimized Vision Resampler initialized:")
        logger.info(f"  - Half precision: {self.use_half_precision}")
        logger.info(f"  - Gradient checkpointing: {self.use_gradient_checkpointing}")
    
    def forward(
        self, 
        vision_features: torch.Tensor,
        target_tokens: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Mobile-optimized forward pass
        """
        if self.use_gradient_checkpointing and self.training:
            try:
                from torch.utils.checkpoint import checkpoint
                return checkpoint(
                    super().forward, 
                    vision_features, 
                    target_tokens, 
                    attention_mask
                )
            except ImportError:
                # Fallback if checkpoint not available
                return super().forward(vision_features, target_tokens, attention_mask)
        else:
            return super().forward(vision_features, target_tokens, attention_mask)


def test_vision_resampler():
    """Test Vision Resampler implementation"""
    logger.info("Testing Vision Resampler...")
    
    # Test parameters
    batch_size = 2
    seq_len = 197  # ViT patch count for 224x224 image
    input_dim = 768
    output_dim = 768
    num_tokens = 64
    
    # Create test input
    vision_features = torch.randn(batch_size, seq_len, input_dim)
    
    # Test basic Vision Resampler
    logger.info("Testing basic Vision Resampler...")
    resampler = VisionResampler(
        input_dim=input_dim,
        output_dim=output_dim,
        num_tokens=num_tokens,
        mobile_optimized=True
    )
    
    output = resampler(vision_features)
    logger.info(f"Input shape: {vision_features.shape}")
    logger.info(f"Output shape: {output.shape}")
    assert output.shape == (batch_size, num_tokens, output_dim)
    
    # Test Adaptive Vision Resampler
    logger.info("Testing Adaptive Vision Resampler...")
    adaptive_resampler = AdaptiveVisionResampler(
        input_dim=input_dim,
        output_dim=output_dim,
        min_tokens=32,
        max_tokens=128,
        mobile_optimized=True
    )
    
    output = adaptive_resampler(vision_features)
    logger.info(f"Adaptive output shape: {output.shape}")
    assert output.shape[0] == batch_size
    assert output.shape[2] == output_dim
    
    # Test Mobile-optimized Vision Resampler
    logger.info("Testing Mobile-optimized Vision Resampler...")
    mobile_resampler = MobileOptimizedVisionResampler(
        input_dim=input_dim,
        output_dim=output_dim,
        num_tokens=num_tokens
    )
    
    output = mobile_resampler(vision_features)
    logger.info(f"Mobile output shape: {output.shape}")
    assert output.shape[0] == batch_size
    assert output.shape[2] == output_dim
    assert output.shape[1] <= num_tokens  # Adaptive can use fewer tokens
    
    logger.info("✅ All tests passed!")


if __name__ == "__main__":
    test_vision_resampler()
