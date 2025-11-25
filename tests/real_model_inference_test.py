#!/usr/bin/env python3
"""
실제 Mobile VLA 모델 추론 테스트
Jetson 환경에서 CUDA 지원 PyTorch를 사용한 실제 모델 추론 테스트
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import time
import os
import sys
from typing import Optional, Tuple, Dict, Any
import json

# Mobile VLA 모델 클래스 정의 (실제 학습 코드와 동일)
class SimpleCLIPLSTMModel(nn.Module):
    """Simple CLIP + LSTM 모델 (실제 학습 코드와 동일)"""
    
    def __init__(self, 
                 vision_dim: int = 2048,
                 text_dim: int = 2048,
                 hidden_dim: int = 4096,
                 action_dim: int = 2,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.num_layers = num_layers
        
        # Vision Encoder (CLIP features)
        self.vision_encoder = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Text Encoder (Kosmos2 features)
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size=hidden_dim * 2,  # vision + text
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Action Predictor
        self.action_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, action_dim)
        )
        
    def forward(self, vision_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """순전파"""
        batch_size = vision_features.size(0)
        
        # Vision encoding
        vision_encoded = self.vision_encoder(vision_features)  # (batch_size, hidden_dim)
        
        # Text encoding
        text_encoded = self.text_encoder(text_features)  # (batch_size, hidden_dim)
        
        # Concatenate vision and text features
        combined_features = torch.cat([vision_encoded, text_encoded], dim=-1)  # (batch_size, hidden_dim * 2)
        
        # Reshape for LSTM (sequence length = 1)
        combined_features = combined_features.unsqueeze(1)  # (batch_size, 1, hidden_dim * 2)
        
        # LSTM processing
        lstm_out, _ = self.lstm(combined_features)  # (batch_size, 1, hidden_dim)
        
        # Get the last output
        lstm_out = lstm_out[:, -1, :]  # (batch_size, hidden_dim)
        
        # Action prediction
        actions = self.action_predictor(lstm_out)  # (batch_size, action_dim)
        
        return actions

class MobileVLAModelLoader:
    """Mobile VLA 모델 로더"""
    
    def __init__(self, model_dir: str = "./Robo+/Mobile_VLA"):
        self.model_dir = model_dir
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 디바이스: {self.device}")
        if torch.cuda.is_available():
            print(f"🎮 CUDA 디바이스: {torch.cuda.get_device_name(0)}")
        
    def load_model(self, checkpoint_path: Optional[str] = None) -> SimpleCLIPLSTMModel:
        """모델 로드"""
        print(f"🚀 Mobile VLA 모델 로딩 중...")
        print(f"📁 모델 디렉토리: {self.model_dir}")
        
        # 체크포인트 경로 자동 탐지
        if checkpoint_path is None:
            checkpoint_path = self._find_best_checkpoint()
        
        if checkpoint_path is None:
            print("❌ 체크포인트 파일을 찾을 수 없습니다.")
            self._list_available_checkpoints()
            return None
        
        print(f"📦 체크포인트 경로: {checkpoint_path}")
        
        try:
            # 모델 생성
            self.model = SimpleCLIPLSTMModel()
            self.model = self.model.to(self.device)
            
            # 체크포인트 로드
            print("📥 체크포인트 로딩 중...")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # 모델 상태 로드
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print("✅ 모델 상태 로드 완료")
            else:
                # 체크포인트가 모델 상태 딕셔너리인 경우
                self.model.load_state_dict(checkpoint)
                print("✅ 모델 상태 로드 완료 (직접 딕셔너리)")
            
            # 모델을 평가 모드로 설정
            self.model.eval()
            
            # 모델 정보 출력
            self._print_model_info(checkpoint)
            
            print("✅ Mobile VLA 모델 로딩 완료!")
            return self.model
            
        except Exception as e:
            print(f"❌ 모델 로딩 실패: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _find_best_checkpoint(self) -> Optional[str]:
        """최고 성능 체크포인트 자동 탐지"""
        possible_paths = [
            f"{self.model_dir}/simple_clip_lstm_model/best_simple_clip_lstm_model.pth",
            f"{self.model_dir}/results/simple_clip_lstm_results_extended/best_simple_clip_lstm_model.pth",
            "./mobile-vla-omniwheel/best_simple_lstm_model.pth"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def _list_available_checkpoints(self):
        """사용 가능한 체크포인트 목록 출력"""
        print("📋 사용 가능한 체크포인트:")
        os.system('find . -name "*.pth" -type f | head -10')
    
    def _print_model_info(self, checkpoint: Dict[str, Any]):
        """모델 정보 출력"""
        print(f"📊 모델 파라미터 수: {sum(p.numel() for p in self.model.parameters()):,}")
        
        if 'epoch' in checkpoint:
            print(f"📈 훈련 에포크: {checkpoint['epoch']}")
        if 'loss' in checkpoint:
            print(f"📉 손실값: {checkpoint['loss']:.4f}")
        if 'val_mae' in checkpoint:
            print(f"📊 검증 MAE: {checkpoint['val_mae']:.4f}")

def test_model_inference():
    """모델 추론 테스트"""
    print("=" * 60)
    print("🧠 실제 Mobile VLA 모델 추론 테스트")
    print("=" * 60)
    
    # 모델 로더 생성
    loader = MobileVLAModelLoader()
    
    # 모델 로드
    model = loader.load_model()
    if model is None:
        print("❌ 모델 로딩 실패")
        return False
    
    print("\n" + "=" * 40)
    print("🔬 추론 성능 테스트")
    print("=" * 40)
    
    # 테스트 데이터 생성 (실제 특징 벡터 크기)
    batch_size = 1
    vision_dim = 2048
    text_dim = 2048
    
    # 랜덤 특징 벡터 생성 (실제 CLIP/Kosmos2 출력과 유사)
    vision_features = torch.randn(batch_size, vision_dim).to(loader.device)
    text_features = torch.randn(batch_size, text_dim).to(loader.device)
    
    print(f"📥 Vision 특징 크기: {vision_features.shape}")
    print(f"📥 Text 특징 크기: {text_features.shape}")
    
    # 워밍업
    print("🔥 워밍업 중...")
    with torch.no_grad():
        for _ in range(10):
            _ = model(vision_features, text_features)
    
    # 추론 시간 측정
    num_runs = 100
    times = []
    
    print(f"⏱️ {num_runs}회 추론 시간 측정 중...")
    with torch.no_grad():
        for i in range(num_runs):
            start_time = time.time()
            output = model(vision_features, text_features)
            end_time = time.time()
            times.append(end_time - start_time)
    
    # 결과 분석
    avg_time = np.mean(times)
    min_time = np.min(times)
    max_time = np.max(times)
    fps = 1.0 / avg_time
    
    print(f"📤 출력 크기: {output.shape}")
    print(f"⏱️ 평균 추론 시간: {avg_time*1000:.2f} ms")
    print(f"⚡ 최소 추론 시간: {min_time*1000:.2f} ms")
    print(f"🐌 최대 추론 시간: {max_time*1000:.2f} ms")
    print(f"🚀 추론 FPS: {fps:.1f}")
    
    # 액션 값 출력
    print(f"🎯 예측 액션: {output.cpu().numpy()}")
    
    return True

def test_memory_usage():
    """메모리 사용량 테스트"""
    print("\n" + "=" * 40)
    print("💾 메모리 사용량 테스트")
    print("=" * 40)
    
    if torch.cuda.is_available():
        # 초기 메모리
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated(0)
        print(f"🔧 초기 메모리: {initial_memory / 1024**2:.1f} MB")
        
        # 모델 로드 후 메모리
        loader = MobileVLAModelLoader()
        model = loader.load_model()
        
        if model is not None:
            model_memory = torch.cuda.memory_allocated(0)
            print(f"🧠 모델 메모리: {model_memory / 1024**2:.1f} MB")
            print(f"📊 모델 메모리 증가: {(model_memory - initial_memory) / 1024**2:.1f} MB")
            
            # 추론 후 메모리
            vision_features = torch.randn(1, 2048).to(loader.device)
            text_features = torch.randn(1, 2048).to(loader.device)
            
            with torch.no_grad():
                _ = model(vision_features, text_features)
            
            inference_memory = torch.cuda.memory_allocated(0)
            print(f"🔬 추론 메모리: {inference_memory / 1024**2:.1f} MB")
            print(f"📈 추론 메모리 증가: {(inference_memory - model_memory) / 1024**2:.1f} MB")
            
            # 총 메모리 사용량
            total_memory = torch.cuda.get_device_properties(0).total_memory
            used_memory = inference_memory
            memory_usage_percent = (used_memory / total_memory) * 100
            
            print(f"💾 총 메모리: {total_memory / 1024**2:.1f} MB")
            print(f"📊 사용률: {memory_usage_percent:.1f}%")
            
            return True
    else:
        print("❌ CUDA를 사용할 수 없습니다.")
        return False

def main():
    """메인 함수"""
    print("🚀 실제 Mobile VLA 모델 추론 테스트")
    print("=" * 60)
    
    # 1. 모델 추론 테스트
    inference_ok = test_model_inference()
    
    # 2. 메모리 사용량 테스트
    memory_ok = test_memory_usage()
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("📊 테스트 결과 요약")
    print("=" * 60)
    
    results = {
        "모델 추론": "✅" if inference_ok else "❌",
        "메모리 사용": "✅" if memory_ok else "❌"
    }
    
    for test_name, result in results.items():
        print(f"{test_name}: {result}")
    
    all_passed = all([inference_ok, memory_ok])
    
    if all_passed:
        print("\n🎉 모든 테스트 통과! 실제 Mobile VLA 모델이 정상 작동합니다.")
        print("\n📋 다음 단계:")
        print("   1. 실제 이미지 입력 처리")
        print("   2. 카메라 스트림 연동")
        print("   3. 실시간 추론 파이프라인 구성")
    else:
        print("\n⚠️ 일부 테스트 실패. 모델 로딩을 확인해주세요.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
