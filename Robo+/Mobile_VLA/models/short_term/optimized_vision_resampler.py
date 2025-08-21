#!/usr/bin/env python3
"""
🚀 Case 2: 단기 적용 - 최적화된 Vision Resampler
목표: MAE 0.5 → 0.3, 정확도 15% → 35%
특징: Vision Resampler 최적화 (latents 64→16, heads 8→4)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedVisionResampler(nn.Module):
    """
    최적화된 Vision Resampler
    - latents 수: 64 → 16 (75% 감소)
    - attention heads: 8 → 4 (50% 감소)
    - FFN 크기: 2x → 1.5x (25% 감소)
    - 적은 데이터셋에 맞게 최적화
    """
    
    def __init__(self, input_dim, output_dim, num_latents=16, num_heads=4, dropout=0.3):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_latents = num_latents
        self.num_heads = num_heads
        self.dropout = dropout
        
        # 최적화된 파라미터
        self.latents = nn.Parameter(torch.randn(num_latents, output_dim))
        self.input_proj = nn.Linear(input_dim, output_dim)
        self.output_proj = nn.Linear(output_dim, output_dim)
        
        # 최적화된 attention (heads 8→4)
        self.attention = nn.MultiheadAttention(
            embed_dim=output_dim, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        
        # 정규화 레이어
        self.norm1 = nn.LayerNorm(output_dim)
        self.norm2 = nn.LayerNorm(output_dim)
        
        # 최적화된 FFN (2x → 1.5x)
        ffn_hidden_dim = int(output_dim * 1.5)
        self.ffn = nn.Sequential(
            nn.Linear(output_dim, ffn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden_dim, output_dim)
        )
        
        # 추가 정규화
        self.final_norm = nn.LayerNorm(output_dim)
        
        logger.info(f"✅ Optimized Vision Resampler 초기화 완료:")
        logger.info(f"   - input_dim: {input_dim}")
        logger.info(f"   - output_dim: {output_dim}")
        logger.info(f"   - num_latents: {num_latents} (64→16, 75% 감소)")
        logger.info(f"   - num_heads: {num_heads} (8→4, 50% 감소)")
        logger.info(f"   - ffn_hidden_dim: {ffn_hidden_dim} (2x→1.5x, 25% 감소)")
        logger.info(f"   - dropout: {dropout}")
    
    def forward(self, x):
        """
        순전파
        Args:
            x: 입력 특징 [batch_size, seq_len, input_dim]
        Returns:
            output: 압축된 특징 [batch_size, output_dim]
        """
        batch_size = x.shape[0]
        
        # 입력 프로젝션
        x = self.input_proj(x)  # [batch_size, seq_len, output_dim]
        
        # 학습 가능한 latents 확장
        latents = self.latents.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, num_latents, output_dim]
        
        # Cross-attention: latents가 query, x가 key/value
        attn_out, _ = self.attention(latents, x, x)
        latents = self.norm1(latents + attn_out)
        
        # Self-attention: latents 간의 attention
        self_attn_out, _ = self.attention(latents, latents, latents)
        latents = self.norm2(latents + self_attn_out)
        
        # Feed-forward network
        ffn_out = self.ffn(latents)
        latents = latents + ffn_out
        
        # 최종 출력 프로젝션
        output = self.output_proj(latents)
        output = self.final_norm(output)
        
        # 평균 풀링으로 압축
        output = output.mean(dim=1)  # [batch_size, output_dim]
        
        return output

class OptimizedVisionResamplerWithSkip(nn.Module):
    """
    Skip connection이 추가된 최적화된 Vision Resampler
    - 정보 손실 최소화
    - 더 안정적인 훈련
    """
    
    def __init__(self, input_dim, output_dim, num_latents=16, num_heads=4, dropout=0.3):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_latents = num_latents
        
        # 기본 resampler
        self.resampler = OptimizedVisionResampler(
            input_dim=input_dim,
            output_dim=output_dim,
            num_latents=num_latents,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Skip connection을 위한 프로젝션
        self.skip_proj = nn.Linear(input_dim, output_dim)
        
        # Skip connection 가중치 (학습 가능)
        self.skip_weight = nn.Parameter(torch.tensor(0.5))
        
        logger.info(f"✅ Optimized Vision Resampler with Skip 초기화 완료:")
        logger.info(f"   - skip_weight: 학습 가능한 파라미터")
    
    def forward(self, x):
        """
        Skip connection이 포함된 순전파
        """
        # 원본 특징의 평균 (skip connection)
        original_mean = x.mean(dim=1)  # [batch_size, input_dim]
        skip_features = self.skip_proj(original_mean)  # [batch_size, output_dim]
        
        # Resampler 출력
        resampled_features = self.resampler(x)  # [batch_size, output_dim]
        
        # Skip connection 결합
        output = self.skip_weight * resampled_features + (1 - self.skip_weight) * skip_features
        
        return output

def test_optimized_vision_resampler():
    """최적화된 Vision Resampler 테스트"""
    
    logger.info("🧪 Optimized Vision Resampler 테스트 시작...")
    
    # 테스트 파라미터
    batch_size = 4
    seq_len = 100
    input_dim = 1024
    output_dim = 256
    num_latents = 16
    num_heads = 4
    
    # 모델 생성
    resampler = OptimizedVisionResampler(
        input_dim=input_dim,
        output_dim=output_dim,
        num_latents=num_latents,
        num_heads=num_heads,
        dropout=0.3
    )
    
    resampler_with_skip = OptimizedVisionResamplerWithSkip(
        input_dim=input_dim,
        output_dim=output_dim,
        num_latents=num_latents,
        num_heads=num_heads,
        dropout=0.3
    )
    
    # 테스트 입력
    x = torch.randn(batch_size, seq_len, input_dim)
    
    # 순전파 테스트
    with torch.no_grad():
        output1 = resampler(x)
        output2 = resampler_with_skip(x)
    
    # 결과 확인
    logger.info(f"📊 테스트 결과:")
    logger.info(f"   - 입력 크기: {x.shape}")
    logger.info(f"   - 기본 출력 크기: {output1.shape}")
    logger.info(f"   - Skip 출력 크기: {output2.shape}")
    logger.info(f"   - 압축 비율: {seq_len} → {num_latents} ({(1-num_latents/seq_len)*100:.1f}% 감소)")
    
    # 파라미터 수 계산
    total_params = sum(p.numel() for p in resampler.parameters())
    trainable_params = sum(p.numel() for p in resampler.parameters() if p.requires_grad)
    
    logger.info(f"📈 파라미터 정보:")
    logger.info(f"   - 총 파라미터: {total_params:,}")
    logger.info(f"   - 훈련 가능 파라미터: {trainable_params:,}")
    logger.info(f"   - 모델 크기: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    # 기존 Vision Resampler와 비교
    original_latents = 64
    original_heads = 8
    original_ffn_ratio = 2.0
    
    original_params = (
        original_latents * output_dim +  # latents
        input_dim * output_dim +  # input_proj
        output_dim * output_dim +  # output_proj
        output_dim * output_dim * 3 +  # attention (Q, K, V)
        output_dim * (output_dim * original_ffn_ratio) * 2  # FFN
    )
    
    compression_ratio = original_params / total_params
    logger.info(f"📊 압축 효율성:")
    logger.info(f"   - 기존 파라미터 (추정): {original_params:,}")
    logger.info(f"   - 최적화 파라미터: {total_params:,}")
    logger.info(f"   - 압축 비율: {compression_ratio:.2f}x")
    logger.info(f"   - 파라미터 감소: {(1-1/compression_ratio)*100:.1f}%")
    
    logger.info("✅ 테스트 완료!")

if __name__ == "__main__":
    test_optimized_vision_resampler()
