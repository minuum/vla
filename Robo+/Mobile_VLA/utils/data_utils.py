#!/usr/bin/env python3
"""
Mobile VLA Data Utilities
RoboVLMs의 claw matrix 및 데이터 처리 유틸리티를 Mobile VLA에 적용
"""

import torch
from einops import repeat
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def claw_matrix(n: int, k: int, device: str = "cpu") -> torch.Tensor:
    """
    Claw matrix 생성 (RoboVLMs에서 가져옴)
    
    Args:
        n: Matrix 크기
        k: 대각선 오프셋
        device: 텐서 디바이스
        
    Returns:
        Claw matrix [n, n]
    """
    upper_triangle_matrix = torch.triu(torch.ones(n, n), diagonal=0).to(device)
    lower_triangle_matrix = torch.tril(torch.ones(n, n), diagonal=k).to(device)

    claw = upper_triangle_matrix * lower_triangle_matrix

    return claw


def generate_chunk_data(
    data: torch.Tensor, 
    window_size: int, 
    chunk_size: int
) -> torch.Tensor:
    """
    Chunk 데이터 생성 (RoboVLMs에서 가져옴, Mobile VLA용으로 수정)
    
    Args:
        data: 입력 데이터 [B, T, ...]
        window_size: 윈도우 크기 
        chunk_size: 청크 크기
        
    Returns:
        청크된 데이터 [B, window_size, chunk_size, ...]
    """
    if data is None:
        return None
        
    bs, seq_len = data.shape[:2]
    raw_data_shape = data.shape[2:]
    data_flatten = data.flatten().view(bs, seq_len, -1)
    
    assert (
        seq_len == window_size + chunk_size
    ), f"시퀀스 길이는 {window_size + chunk_size}여야 함, 실제: {seq_len}"
    
    data_flatten = repeat(data_flatten, "b s d -> b w s d", w=window_size)

    mask = claw_matrix(seq_len, chunk_size - 1, data_flatten.device)
    # mask = mask - torch.diag_embed(mask.diag()) # 현재 관찰 마스크를 0으로 설정
    mask = mask[:window_size].bool()

    mask = repeat(mask, "w s -> b w s d", b=bs, d=data_flatten.shape[-1])
    data_flatten = torch.masked_select(data_flatten, mask)

    data_flatten = data_flatten.view(bs, window_size, chunk_size, *raw_data_shape)

    return data_flatten


def mobile_vla_sequence_chunking(
    images: torch.Tensor,
    actions: torch.Tensor, 
    events: torch.Tensor,
    window_size: int = 10,
    chunk_size: int = 8
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Mobile VLA 시퀀스 청킹 (claw matrix 적용)
    
    Args:
        images: [B, T, C, H, W] 이미지 시퀀스
        actions: [B, T, 3] 액션 시퀀스 
        events: [B, T] 이벤트 시퀀스
        window_size: 윈도우 크기
        chunk_size: 청크 크기
        
    Returns:
        청킹된 images, actions, events
    """
    logger.info(f"🔄 Mobile VLA 시퀀스 청킹: window={window_size}, chunk={chunk_size}")
    
    # 이미지 청킹
    chunked_images = generate_chunk_data(images, window_size, chunk_size)
    logger.info(f"   Images: {images.shape} → {chunked_images.shape}")
    
    # 액션 청킹  
    chunked_actions = generate_chunk_data(actions, window_size, chunk_size)
    logger.info(f"   Actions: {actions.shape} → {chunked_actions.shape}")
    
    # 이벤트 청킹
    chunked_events = generate_chunk_data(events, window_size, chunk_size)
    logger.info(f"   Events: {events.shape} → {chunked_events.shape}")
    
    return chunked_images, chunked_actions, chunked_events


def test_claw_matrix():
    """Claw matrix 테스트"""
    print("🧪 Claw Matrix 테스트")
    
    # 기본 테스트
    n, k = 5, 2
    claw = claw_matrix(n, k)
    print(f"Claw Matrix ({n}x{n}, k={k}):")
    print(claw)
    
    # Mobile VLA 시퀀스 길이 테스트 (18 프레임)
    n, k = 18, 8
    claw = claw_matrix(n, k)
    print(f"\nMobile VLA Claw Matrix ({n}x{n}, k={k}):")
    print(f"Shape: {claw.shape}")
    print(f"Non-zero elements: {claw.sum().item()}")
    
    # 청킹 테스트
    print(f"\n🔄 청킹 테스트")
    B, T = 2, 18
    window_size, chunk_size = 10, 8
    
    # 더미 데이터
    images = torch.randn(B, T, 3, 224, 224)
    actions = torch.randn(B, T, 3)
    events = torch.randint(0, 3, (B, T))
    
    try:
        chunked_images, chunked_actions, chunked_events = mobile_vla_sequence_chunking(
            images, actions, events, window_size, chunk_size
        )
        print("✅ 청킹 성공!")
        
    except Exception as e:
        print(f"❌ 청킹 실패: {e}")


if __name__ == "__main__":
    test_claw_matrix()
