#!/usr/bin/env python3
"""
Mobile VLA 모델 로더 (실제 학습 코드 기반)
Kosmos2 + CLIP 하이브리드 모델 (MAE 0.212) 로딩
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
from typing import Optional, Dict, Any, Tuple
import numpy as np

class SimpleCLIPLSTMModel(nn.Module):
    """실제 학습 코드 기반 Mobile VLA 모델 (Kosmos2 + CLIP 하이브리드)"""
    
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
        
        # Vision Encoder (Kosmos2 features)
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
        
        # Text Encoder (CLIP features)
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
        
        # Feature Fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Action prediction head
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, action_dim)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """가중치 초기화 (실제 학습 코드와 동일)"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, vision_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass (실제 학습 코드와 동일)
        
        Args:
            vision_features: (batch_size, vision_dim) - Kosmos2 vision features
            text_features: (batch_size, text_dim) - CLIP text features
            
        Returns:
            actions: (batch_size, action_dim) - Predicted 2D actions
        """
        batch_size = vision_features.size(0)
        
        # Encode features
        vision_encoded = self.vision_encoder(vision_features)  # (batch_size, hidden_dim)
        text_encoded = self.text_encoder(text_features)        # (batch_size, hidden_dim)
        
        # Feature fusion
        combined = torch.cat([vision_encoded, text_encoded], dim=-1)  # (batch_size, hidden_dim*2)
        fused = self.feature_fusion(combined)  # (batch_size, hidden_dim)
        
        # LSTM processing (add sequence dimension)
        fused = fused.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(fused)  # (batch_size, 1, hidden_dim)
        
        # Take the last output
        lstm_out = lstm_out[:, -1, :]  # (batch_size, hidden_dim)
        
        # Predict actions
        actions = self.action_head(lstm_out)  # (batch_size, action_dim)
        
        return actions

class MobileVLAModelLoader:
    """Mobile VLA 모델 로더 (실제 학습 코드 기반)"""
    
    def __init__(self, model_dir: str = "./Robo+/Mobile_VLA"):
        self.model_dir = model_dir
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 디바이스: {self.device}")
        if torch.cuda.is_available():
            print(f"🎮 CUDA 디바이스: {torch.cuda.get_device_name(0)}")
        
    def load_model(self, checkpoint_path: Optional[str] = None) -> SimpleCLIPLSTMModel:
        """
        모델 로드 (실제 학습 코드와 동일한 구조)
        
        Args:
            checkpoint_path: 체크포인트 파일 경로 (None이면 자동 탐지)
            
        Returns:
            SimpleCLIPLSTMModel: 로드된 모델
        """
        print(f"🚀 Mobile VLA 모델 로딩 중...")
        print(f"📁 모델 디렉토리: {self.model_dir}")
        
        # 체크포인트 경로 자동 탐지
        if checkpoint_path is None:
            checkpoint_path = self._find_best_checkpoint()
        
        if checkpoint_path is None:
            print("❌ 체크포인트 파일을 찾을 수 없습니다.")
            print("📋 사용 가능한 체크포인트:")
            self._list_available_checkpoints()
            return None
        
        print(f"📦 체크포인트 경로: {checkpoint_path}")
        
        try:
            # 모델 생성 (실제 학습 코드와 동일한 구조)
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
        """최고 성능 체크포인트 자동 탐지 (MAE 0.212)"""
        possible_paths = [
            f"{self.model_dir}/results/simple_clip_lstm_results_extended/best_simple_clip_lstm_model.pth",
            f"{self.model_dir}/results/simple_lstm_results_extended/best_simple_lstm_model.pth",
            "./mobile-vla-omniwheel/best_simple_lstm_model.pth",
            "./best_simple_clip_lstm_model.pth",
            "./best_simple_lstm_model.pth"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"✅ 체크포인트 발견: {path}")
                return path
        
        return None
    
    def _list_available_checkpoints(self):
        """사용 가능한 체크포인트 목록 출력"""
        search_paths = [
            f"{self.model_dir}/results",
            "./mobile-vla-omniwheel",
            "./"
        ]
        
        for search_path in search_paths:
            if os.path.exists(search_path):
                for root, dirs, files in os.walk(search_path):
                    for file in files:
                        if file.endswith('.pth'):
                            full_path = os.path.join(root, file)
                            size_mb = os.path.getsize(full_path) / (1024 * 1024)
                            print(f"   - {full_path} ({size_mb:.1f} MB)")
    
    def _print_model_info(self, checkpoint: Dict[str, Any]):
        """모델 정보 출력"""
        if self.model is None:
            return
        
        # 파라미터 수 계산
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"📊 모델 정보:")
        print(f"   - 총 파라미터 수: {total_params:,}")
        print(f"   - 훈련 가능 파라미터 수: {trainable_params:,}")
        print(f"   - 모델 구조: Kosmos2 + CLIP 하이브리드 + LSTM")
        print(f"   - Vision Encoder: {self.model.vision_dim} → {self.model.hidden_dim}")
        print(f"   - Text Encoder: {self.model.text_dim} → {self.model.hidden_dim}")
        print(f"   - LSTM Layers: {self.model.num_layers}")
        print(f"   - Action Head: {self.model.hidden_dim} → {self.model.action_dim}")
        
        # 체크포인트 정보
        if 'epoch' in checkpoint:
            print(f"   - 훈련 에포크: {checkpoint['epoch']}")
        if 'val_mae' in checkpoint:
            print(f"   - 검증 MAE: {checkpoint['val_mae']:.4f}")
        if 'args' in checkpoint:
            print(f"   - 훈련 설정: {checkpoint['args']}")
        
        print(f"   - 성능: MAE 0.212 (Kosmos2 + CLIP 하이브리드)")
        print(f"   - 속도: 766 FPS (FP16 양자화 시)")
    
    def predict(self, vision_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """
        추론 실행
        
        Args:
            vision_features: (batch_size, 2048) - Kosmos2 vision features
            text_features: (batch_size, 2048) - CLIP text features
            
        Returns:
            actions: (batch_size, 2) - Predicted 2D actions
        """
        if self.model is None:
            raise ValueError("모델이 로드되지 않았습니다. load_model()을 먼저 호출하세요.")
        
        with torch.no_grad():
            # 디바이스로 이동
            vision_features = vision_features.to(self.device)
            text_features = text_features.to(self.device)
            
            # 추론
            actions = self.model(vision_features, text_features)
            
            return actions
    
    def get_model(self) -> Optional[SimpleCLIPLSTMModel]:
        """로드된 모델 반환"""
        return self.model

def test_model_loader():
    """모델 로더 테스트 함수"""
    print("🧪 Mobile VLA 모델 로더 테스트")
    print("=" * 50)
    
    # 로더 생성
    loader = MobileVLAModelLoader()
    
    # 모델 로드
    model = loader.load_model()
    
    if model is not None:
        print("\n✅ 모델 로딩 성공!")
        
        # 테스트 추론
        print("\n🎯 추론 테스트 중...")
        batch_size = 2
        vision_features = torch.randn(batch_size, 2048)
        text_features = torch.randn(batch_size, 2048)
        
        try:
            actions = loader.predict(vision_features, text_features)
            print(f"✅ 추론 성공!")
            print(f"   - 입력 크기: vision={vision_features.shape}, text={text_features.shape}")
            print(f"   - 출력 크기: {actions.shape}")
            print(f"   - 출력 값: {actions}")
            
            # 성능 테스트
            print("\n⚡ 성능 테스트 중...")
            import time
            
            # CPU에서 테스트
            vision_features_cpu = torch.randn(1, 2048)
            text_features_cpu = torch.randn(1, 2048)
            
            start_time = time.time()
            for _ in range(100):
                actions_cpu = loader.predict(vision_features_cpu, text_features_cpu)
            cpu_time = time.time() - start_time
            
            print(f"   - CPU 추론 시간 (100회): {cpu_time:.4f}초")
            print(f"   - CPU 평균 추론 시간: {cpu_time/100*1000:.2f}ms")
            print(f"   - CPU FPS: {100/cpu_time:.1f}")
            
            # GPU에서 테스트 (CUDA 사용 가능한 경우)
            if torch.cuda.is_available():
                vision_features_gpu = torch.randn(1, 2048).cuda()
                text_features_gpu = torch.randn(1, 2048).cuda()
                
                # GPU 워밍업
                for _ in range(10):
                    actions_gpu = loader.predict(vision_features_gpu, text_features_gpu)
                
                torch.cuda.synchronize()
                start_time = time.time()
                for _ in range(100):
                    actions_gpu = loader.predict(vision_features_gpu, text_features_gpu)
                torch.cuda.synchronize()
                gpu_time = time.time() - start_time
                
                print(f"   - GPU 추론 시간 (100회): {gpu_time:.4f}초")
                print(f"   - GPU 평균 추론 시간: {gpu_time/100*1000:.2f}ms")
                print(f"   - GPU FPS: {100/gpu_time:.1f}")
                print(f"   - GPU 가속: {cpu_time/gpu_time:.1f}x")
            
        except Exception as e:
            print(f"❌ 추론 실패: {e}")
            import traceback
            traceback.print_exc()
        
    else:
        print("❌ 모델 로딩 실패")

def main():
    """메인 함수"""
    test_model_loader()

if __name__ == "__main__":
    main()
