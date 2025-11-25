#!/usr/bin/env python3
"""
Simple Claw Matrix Implementation
Vision-Language 특징 간의 정렬을 개선하는 간단한 Claw Matrix

주요 기능:
1. 간단한 Claw Matrix 계산
2. Vision-Language 정렬 개선
3. 모바일 최적화
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional, Tuple, Dict, Any

logger = logging.getLogger(__name__)

class SimpleClawMatrix(nn.Module):
    """
    Simple Claw Matrix for Vision-Language Alignment
    
    간단하고 효율적인 Claw Matrix 구현
    """
    
    def __init__(
        self,
        feature_dim: int = 768,
        temperature: float = 0.07,
        dropout: float = 0.1
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.temperature = temperature
        
        # Simple projection layers
        self.vision_projection = nn.Linear(feature_dim, feature_dim)
        self.language_projection = nn.Linear(feature_dim, feature_dim)
        
        # Claw matrix parameter
        self.claw_matrix = nn.Parameter(torch.randn(feature_dim, feature_dim))
        nn.init.xavier_uniform_(self.claw_matrix)
        
        # Output projection - handle dynamic input size
        self.output_projection = None  # Will be created dynamically
        
        logger.info(f"Simple Claw Matrix initialized:")
        logger.info(f"  - Feature dim: {feature_dim}")
        logger.info(f"  - Temperature: {temperature}")
    
    def forward(
        self, 
        vision_features: torch.Tensor, 
        language_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with Simple Claw Matrix
        
        Args:
            vision_features: [batch_size, seq_len, feature_dim] Vision features
            language_features: [batch_size, seq_len, feature_dim] Language features
            
        Returns:
            aligned_features: [batch_size, seq_len, feature_dim] Aligned features
        """
        # Project features
        vision_proj = self.vision_projection(vision_features)  # [batch_size, seq_len, feature_dim]
        language_proj = self.language_projection(language_features)  # [batch_size, seq_len, feature_dim]
        
        # Compute similarity matrix
        similarity = torch.bmm(
            vision_proj, 
            language_proj.transpose(-2, -1)
        )  # [batch_size, seq_len, seq_len]
        
        # Apply temperature scaling
        similarity = similarity / self.temperature
        
        # Apply softmax for attention weights
        attention_weights = F.softmax(similarity, dim=-1)
        
        # Apply attention to align features
        aligned_vision = torch.bmm(attention_weights, vision_proj)  # [batch_size, seq_len, feature_dim]
        aligned_language = torch.bmm(attention_weights.transpose(-2, -1), language_proj)  # [batch_size, seq_len, feature_dim]
        
        # Combine aligned features
        combined_features = torch.cat([aligned_vision, aligned_language], dim=-1)  # [batch_size, seq_len, feature_dim*2]
        
        # Output projection - handle dynamic input size
        if self.output_projection is None or combined_features.size(-1) != self.output_projection[0].in_features:
            # Create dynamic projection layer if needed
            self.output_projection = nn.Sequential(
                nn.Linear(combined_features.size(-1), self.feature_dim),
                nn.LayerNorm(self.feature_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ).to(combined_features.device)
        
        output_features = self.output_projection(combined_features)  # [batch_size, seq_len, feature_dim]
        
        return output_features

class MobileOptimizedSimpleClawMatrix(nn.Module):
    """
    Mobile-optimized Simple Claw Matrix
    
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
        
        # Lightweight projections
        self.vision_projection = nn.Linear(feature_dim, feature_dim)
        self.language_projection = nn.Linear(feature_dim, feature_dim)
        
        # Simple output projection - handle dynamic input size
        self.output_projection = None  # Will be created dynamically
        
        # Convert to half precision if supported
        if use_half_precision and torch.cuda.is_available():
            self.half()
        
        logger.info(f"Mobile-optimized Simple Claw Matrix initialized:")
        logger.info(f"  - Feature dim: {feature_dim}")
        logger.info(f"  - Half precision: {use_half_precision}")
        logger.info(f"  - Gradient checkpointing: {use_gradient_checkpointing}")
    
    def forward(
        self, 
        vision_features: torch.Tensor, 
        language_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Mobile-optimized forward pass
        """
        if self.use_gradient_checkpointing and self.training:
            try:
                from torch.utils.checkpoint import checkpoint
                return checkpoint(
                    self._forward_impl, vision_features, language_features, use_reentrant=False
                )
            except ImportError:
                return self._forward_impl(vision_features, language_features)
        else:
            return self._forward_impl(vision_features, language_features)
    
    def _forward_impl(
        self, 
        vision_features: torch.Tensor, 
        language_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Implementation of forward pass
        """
        # Project features
        vision_proj = self.vision_projection(vision_features)  # [batch_size, seq_len, feature_dim]
        language_proj = self.language_projection(language_features)  # [batch_size, seq_len, feature_dim]
        
        # Compute similarity matrix
        similarity = torch.bmm(vision_proj, language_proj.transpose(-2, -1))  # [batch_size, seq_len, seq_len]
        
        # Apply softmax for attention weights
        attention_weights = F.softmax(similarity, dim=-1)
        
        # Apply attention to align features
        aligned_vision = torch.bmm(attention_weights, vision_proj)  # [batch_size, seq_len, feature_dim]
        aligned_language = torch.bmm(attention_weights.transpose(-2, -1), language_proj)  # [batch_size, seq_len, feature_dim]
        
        # Combine and project
        combined = torch.cat([aligned_vision, aligned_language], dim=-1)  # [batch_size, seq_len, feature_dim*2]
        
        # Handle dynamic input size
        if self.output_projection is None or combined.size(-1) != self.output_projection.in_features:
            # Create dynamic projection layer if needed
            self.output_projection = nn.Linear(combined.size(-1), self.feature_dim).to(combined.device)
        
        output = self.output_projection(combined)  # [batch_size, seq_len, feature_dim]
        
        return output

def test_simple_claw_matrix():
    """Test Simple Claw Matrix implementations"""
    print("Testing Simple Claw Matrix implementations...")
    
    # Test data
    batch_size, seq_len, feature_dim = 2, 10, 768
    vision_features = torch.randn(batch_size, seq_len, feature_dim)
    language_features = torch.randn(batch_size, seq_len, feature_dim)
    
    # Test basic Simple Claw Matrix
    print("\n1. Testing SimpleClawMatrix...")
    claw_matrix = SimpleClawMatrix(feature_dim)
    output = claw_matrix(vision_features, language_features)
    print(f"   Input vision shape: {vision_features.shape}")
    print(f"   Input language shape: {language_features.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Output mean: {output.mean().item():.4f}")
    print(f"   Output std: {output.std().item():.4f}")
    
    # Test mobile optimized Simple Claw Matrix
    print("\n2. Testing MobileOptimizedSimpleClawMatrix...")
    mobile_claw = MobileOptimizedSimpleClawMatrix(feature_dim)
    output = mobile_claw(vision_features, language_features)
    print(f"   Input vision shape: {vision_features.shape}")
    print(f"   Input language shape: {language_features.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Output mean: {output.mean().item():.4f}")
    print(f"   Output std: {output.std().item():.4f}")
    
    print("\n✅ All Simple Claw Matrix tests passed!")

if __name__ == "__main__":
    test_simple_claw_matrix()
