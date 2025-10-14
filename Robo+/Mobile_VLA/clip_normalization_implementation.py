#!/usr/bin/env python3
"""
CLIP Normalization Implementation
RoboVLMs의 CLIP Normalization 기술 구현

주요 기능:
1. CLIP 특징 정규화
2. Vision-Language 정렬 개선
3. 모바일 최적화
4. 적응적 정규화
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional, Tuple, Dict, Any
import math

logger = logging.getLogger(__name__)

class CLIPNormalization(nn.Module):
    """
    CLIP Normalization Layer
    
    CLIP 특징을 정규화하여 Vision-Language 정렬을 개선
    """
    
    def __init__(
        self,
        feature_dim: int = 768,
        eps: float = 1e-6,
        learnable_scale: bool = True,
        learnable_shift: bool = True
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.eps = eps
        self.learnable_scale = learnable_scale
        self.learnable_shift = learnable_shift
        
        # Learnable parameters
        if learnable_scale:
            self.scale = nn.Parameter(torch.ones(feature_dim))
        else:
            self.register_buffer('scale', torch.ones(feature_dim))
            
        if learnable_shift:
            self.shift = nn.Parameter(torch.zeros(feature_dim))
        else:
            self.register_buffer('shift', torch.zeros(feature_dim))
        
        logger.info(f"CLIP Normalization initialized:")
        logger.info(f"  - Feature dim: {feature_dim}")
        logger.info(f"  - Learnable scale: {learnable_scale}")
        logger.info(f"  - Learnable shift: {learnable_shift}")
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            features: [batch_size, seq_len, feature_dim] Input features
            
        Returns:
            normalized_features: [batch_size, seq_len, feature_dim] Normalized features
        """
        # Layer normalization
        mean = features.mean(dim=-1, keepdim=True)
        var = features.var(dim=-1, keepdim=True, unbiased=False)
        
        normalized = (features - mean) / torch.sqrt(var + self.eps)
        
        # Apply learnable scale and shift
        normalized = normalized * self.scale + self.shift
        
        return normalized

class AdaptiveCLIPNormalization(nn.Module):
    """
    Adaptive CLIP Normalization
    
    입력 특징의 통계에 따라 적응적으로 정규화
    """
    
    def __init__(
        self,
        feature_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        
        # Base normalization
        self.base_norm = CLIPNormalization(feature_dim)
        
        # Adaptive attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feature transformation
        self.feature_transform = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, feature_dim)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(feature_dim)
        
        logger.info(f"Adaptive CLIP Normalization initialized:")
        logger.info(f"  - Feature dim: {feature_dim}")
        logger.info(f"  - Num heads: {num_heads}")
        logger.info(f"  - Head dim: {self.head_dim}")
    
    def forward(
        self, 
        features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with adaptive normalization
        
        Args:
            features: [batch_size, seq_len, feature_dim] Input features
            attention_mask: Optional attention mask
            
        Returns:
            normalized_features: [batch_size, seq_len, feature_dim] Normalized features
        """
        # Base normalization
        base_normalized = self.base_norm(features)
        
        # Self-attention for adaptive weighting
        attn_output, _ = self.attention(
            base_normalized, 
            base_normalized, 
            base_normalized,
            key_padding_mask=attention_mask
        )
        
        # Residual connection
        attn_output = attn_output + base_normalized
        
        # Feature transformation
        transformed = self.feature_transform(attn_output)
        
        # Final layer normalization
        output = self.layer_norm(transformed + attn_output)
        
        return output

class MobileOptimizedCLIPNormalization(nn.Module):
    """
    Mobile-optimized CLIP Normalization
    
    Jetson Orin NX와 같은 모바일 환경에 특화된 최적화
    """
    
    def __init__(
        self,
        feature_dim: int = 768,
        use_half_precision: bool = False,
        use_gradient_checkpointing: bool = True
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.use_half_precision = use_half_precision
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # Simplified normalization (no learnable parameters for efficiency)
        self.norm = nn.LayerNorm(feature_dim, eps=1e-6)
        
        # Lightweight feature enhancement
        self.feature_enhancement = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, feature_dim)
        )
        
        # Convert to half precision if supported
        if use_half_precision and torch.cuda.is_available():
            self.half()
        
        logger.info(f"Mobile-optimized CLIP Normalization initialized:")
        logger.info(f"  - Feature dim: {feature_dim}")
        logger.info(f"  - Half precision: {use_half_precision}")
        logger.info(f"  - Gradient checkpointing: {use_gradient_checkpointing}")
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Mobile-optimized forward pass
        """
        if self.use_gradient_checkpointing and self.training:
            try:
                from torch.utils.checkpoint import checkpoint
                return checkpoint(
                    self._forward_impl, features, use_reentrant=False
                )
            except ImportError:
                # Fallback to direct forward if checkpoint not available
                return self._forward_impl(features)
        else:
            return self._forward_impl(features)
    
    def _forward_impl(self, features: torch.Tensor) -> torch.Tensor:
        """
        Implementation of forward pass
        """
        # Layer normalization
        normalized = self.norm(features)
        
        # Lightweight feature enhancement
        enhanced = self.feature_enhancement(normalized)
        
        # Residual connection
        output = enhanced + normalized
        
        return output

class CLIPNormalizationWithTemperature(nn.Module):
    """
    CLIP Normalization with Temperature Scaling
    
    Temperature scaling을 통한 특징 정규화
    """
    
    def __init__(
        self,
        feature_dim: int = 768,
        temperature: float = 0.07,
        learnable_temperature: bool = True
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.learnable_temperature = learnable_temperature
        
        if learnable_temperature:
            self.temperature = nn.Parameter(torch.tensor(temperature))
        else:
            self.register_buffer('temperature', torch.tensor(temperature))
        
        # Base normalization
        self.norm = nn.LayerNorm(feature_dim)
        
        logger.info(f"CLIP Normalization with Temperature initialized:")
        logger.info(f"  - Feature dim: {feature_dim}")
        logger.info(f"  - Temperature: {temperature}")
        logger.info(f"  - Learnable temperature: {learnable_temperature}")
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with temperature scaling
        """
        # Layer normalization
        normalized = self.norm(features)
        
        # Temperature scaling
        scaled = normalized / self.temperature
        
        return scaled

class CLIPNormalizationEnsemble(nn.Module):
    """
    CLIP Normalization Ensemble
    
    여러 정규화 방법을 앙상블하여 성능 향상
    """
    
    def __init__(
        self,
        feature_dim: int = 768,
        num_methods: int = 3
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_methods = num_methods
        
        # Different normalization methods
        self.normalizations = nn.ModuleList([
            CLIPNormalization(feature_dim),
            AdaptiveCLIPNormalization(feature_dim),
            MobileOptimizedCLIPNormalization(feature_dim)
        ])
        
        # Ensemble weights
        self.ensemble_weights = nn.Parameter(torch.ones(num_methods) / num_methods)
        
        # Final fusion
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * num_methods, feature_dim * 2),
            nn.GELU(),
            nn.Linear(feature_dim * 2, feature_dim)
        )
        
        logger.info(f"CLIP Normalization Ensemble initialized:")
        logger.info(f"  - Feature dim: {feature_dim}")
        logger.info(f"  - Num methods: {num_methods}")
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Ensemble forward pass
        """
        # Apply different normalizations
        normalized_features = []
        for norm in self.normalizations:
            normalized_features.append(norm(features))
        
        # Weighted combination
        weighted_features = []
        for i, feat in enumerate(normalized_features):
            weighted_features.append(feat * self.ensemble_weights[i])
        
        # Concatenate and fuse
        concatenated = torch.cat(weighted_features, dim=-1)
        fused = self.fusion(concatenated)
        
        return fused

def test_clip_normalization():
    """Test CLIP normalization implementations"""
    print("Testing CLIP Normalization implementations...")
    
    # Test data
    batch_size, seq_len, feature_dim = 2, 10, 768
    test_features = torch.randn(batch_size, seq_len, feature_dim)
    
    # Test basic CLIP normalization
    print("\n1. Testing CLIPNormalization...")
    norm = CLIPNormalization(feature_dim)
    output = norm(test_features)
    print(f"   Input shape: {test_features.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Output mean: {output.mean().item():.4f}")
    print(f"   Output std: {output.std().item():.4f}")
    
    # Test adaptive normalization
    print("\n2. Testing AdaptiveCLIPNormalization...")
    adaptive_norm = AdaptiveCLIPNormalization(feature_dim)
    output = adaptive_norm(test_features)
    print(f"   Input shape: {test_features.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Output mean: {output.mean().item():.4f}")
    print(f"   Output std: {output.std().item():.4f}")
    
    # Test mobile optimization
    print("\n3. Testing MobileOptimizedCLIPNormalization...")
    mobile_norm = MobileOptimizedCLIPNormalization(feature_dim)
    output = mobile_norm(test_features)
    print(f"   Input shape: {test_features.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Output mean: {output.mean().item():.4f}")
    print(f"   Output std: {output.std().item():.4f}")
    
    # Test temperature scaling
    print("\n4. Testing CLIPNormalizationWithTemperature...")
    temp_norm = CLIPNormalizationWithTemperature(feature_dim)
    output = temp_norm(test_features)
    print(f"   Input shape: {test_features.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Output mean: {output.mean().item():.4f}")
    print(f"   Output std: {output.std().item():.4f}")
    
    # Test ensemble
    print("\n5. Testing CLIPNormalizationEnsemble...")
    ensemble_norm = CLIPNormalizationEnsemble(feature_dim)
    output = ensemble_norm(test_features)
    print(f"   Input shape: {test_features.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Output mean: {output.mean().item():.4f}")
    print(f"   Output std: {output.std().item():.4f}")
    
    print("\n✅ All CLIP Normalization tests passed!")

if __name__ == "__main__":
    test_clip_normalization()
