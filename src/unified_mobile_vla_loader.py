#!/usr/bin/env python3
"""
통합 Mobile VLA 모델 로더
MODEL_RANKING.md 기반으로 모든 모델을 지원하는 통합 로더
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import json
from typing import Optional, Dict, Any, Tuple, Union
import numpy as np
from enum import Enum

class ModelType(Enum):
    """지원하는 모델 타입"""
    KOSMOS2_CLIP_HYBRID = "kosmos2_clip_hybrid"  # MAE 0.212 (1위)
    KOSMOS2_PURE = "kosmos2_pure"                # MAE 0.222 (2위)

class UnifiedMobileVLAModel(nn.Module):
    """통합 Mobile VLA 모델 (모든 모델 타입 지원)"""
    
    def __init__(self, 
                 model_type: ModelType = ModelType.KOSMOS2_CLIP_HYBRID,
                 vision_dim: int = 2048,
                 text_dim: int = 2048,
                 hidden_dim: int = 4096,
                 action_dim: int = 2,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        
        self.model_type = model_type
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
        
        # Text Encoder (CLIP features) - 하이브리드 모델에서만 사용
        if model_type == ModelType.KOSMOS2_CLIP_HYBRID:
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
            
            # Feature Fusion (하이브리드 모델)
            self.feature_fusion = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        else:
            # 순수 Kosmos2 모델은 text_encoder 없음
            self.text_encoder = None
            self.feature_fusion = None
        
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
        """가중치 초기화"""
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
    
    def forward(self, vision_features: torch.Tensor, text_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            vision_features: (batch_size, vision_dim) - Kosmos2 vision features
            text_features: (batch_size, text_dim) - CLIP text features (하이브리드 모델에서만)
            
        Returns:
            actions: (batch_size, action_dim) - Predicted 2D actions
        """
        batch_size = vision_features.size(0)
        
        # Encode vision features
        vision_encoded = self.vision_encoder(vision_features)  # (batch_size, hidden_dim)
        
        if self.model_type == ModelType.KOSMOS2_CLIP_HYBRID:
            # 하이브리드 모델: Vision + Text fusion
            if text_features is None:
                raise ValueError("하이브리드 모델에서는 text_features가 필요합니다.")
            
            text_encoded = self.text_encoder(text_features)  # (batch_size, hidden_dim)
            combined = torch.cat([vision_encoded, text_encoded], dim=-1)  # (batch_size, hidden_dim*2)
            fused = self.feature_fusion(combined)  # (batch_size, hidden_dim)
        else:
            # 순수 Kosmos2 모델: Vision만 사용
            fused = vision_encoded  # (batch_size, hidden_dim)
        
        # LSTM processing
        fused = fused.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        lstm_out, (hidden, cell) = self.lstm(fused)  # (batch_size, 1, hidden_dim)
        lstm_out = lstm_out[:, -1, :]  # (batch_size, hidden_dim)
        
        # Predict actions
        actions = self.action_head(lstm_out)  # (batch_size, action_dim)
        
        return actions

class UnifiedMobileVLAModelLoader:
    """통합 Mobile VLA 모델 로더 (모든 모델 타입 지원)"""
    
    def __init__(self, model_dir: str = "./Robo+/Mobile_VLA"):
        self.model_dir = model_dir
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = None
        
        print(f"🔧 디바이스: {self.device}")
        if torch.cuda.is_available():
            print(f"🎮 CUDA 디바이스: {torch.cuda.get_device_name(0)}")
        
        # 모델 정보 (MODEL_RANKING.md 기반)
        self.model_info = {
            ModelType.KOSMOS2_CLIP_HYBRID: {
                'name': 'Kosmos2 + CLIP 하이브리드',
                'mae': 0.212,
                'fps': 765.7,
                'params': 1859579651,
                'size_gb': 7.8,
                'rank': 1,
                'checkpoint_path': f"{model_dir}/results/simple_clip_lstm_results_extended/best_simple_clip_lstm_model.pth",
                'training_file': f"{model_dir}/models/core/train_simple_clip_lstm_core.py"
            },
            ModelType.KOSMOS2_PURE: {
                'name': '순수 Kosmos2',
                'mae': 0.222,
                'fps': 755.2,
                'params': 1703973122,
                'size_gb': 7.1,
                'rank': 2,
                'checkpoint_path': "./mobile-vla-omniwheel/best_simple_lstm_model.pth",
                'training_file': f"{model_dir}/models/core/train_simple_lstm_core.py"
            }
        }
    
    def list_available_models(self):
        """사용 가능한 모델 목록 출력"""
        print("📋 사용 가능한 모델 목록:")
        print("=" * 60)
        
        for model_type, info in self.model_info.items():
            status = "✅" if os.path.exists(info['checkpoint_path']) else "❌"
            print(f"{status} {info['rank']}위: {info['name']}")
            print(f"   - MAE: {info['mae']}")
            print(f"   - FPS: {info['fps']}")
            print(f"   - 파라미터: {info['params']:,}")
            print(f"   - 크기: {info['size_gb']}GB")
            print(f"   - 체크포인트: {info['checkpoint_path']}")
            print(f"   - 존재: {os.path.exists(info['checkpoint_path'])}")
            print()
    
    def load_model(self, model_type: ModelType = ModelType.KOSMOS2_CLIP_HYBRID, 
                   checkpoint_path: Optional[str] = None) -> UnifiedMobileVLAModel:
        """
        모델 로드
        
        Args:
            model_type: 로드할 모델 타입
            checkpoint_path: 체크포인트 파일 경로 (None이면 자동 탐지)
            
        Returns:
            UnifiedMobileVLAModel: 로드된 모델
        """
        self.model_type = model_type
        model_info = self.model_info[model_type]
        
        print(f"🚀 {model_info['name']} 모델 로딩 중...")
        print(f"📁 모델 디렉토리: {self.model_dir}")
        print(f"🎯 목표 성능: MAE {model_info['mae']}, FPS {model_info['fps']}")
        
        # 체크포인트 경로 결정
        if checkpoint_path is None:
            checkpoint_path = model_info['checkpoint_path']
        
        if not os.path.exists(checkpoint_path):
            print(f"❌ 체크포인트 파일이 존재하지 않습니다: {checkpoint_path}")
            print("📋 사용 가능한 체크포인트:")
            self._list_available_checkpoints()
            return None
        
        print(f"📦 체크포인트 경로: {checkpoint_path}")
        
        try:
            # 모델 생성
            self.model = UnifiedMobileVLAModel(model_type=model_type)
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
            self._print_model_info(checkpoint, model_info)
            
            print("✅ 모델 로딩 완료!")
            return self.model
            
        except Exception as e:
            print(f"❌ 모델 로딩 실패: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _list_available_checkpoints(self):
        """사용 가능한 체크포인트 목록 출력"""
        search_paths = [
            f"{self.model_dir}/results",
            "./mobile-vla-omniwheel",
            "./vla/mobile-vla-omniwheel",
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
    
    def _print_model_info(self, checkpoint: Dict[str, Any], model_info: Dict[str, Any]):
        """모델 정보 출력"""
        if self.model is None:
            return
        
        # 파라미터 수 계산
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"📊 모델 정보:")
        print(f"   - 모델 타입: {model_info['name']}")
        print(f"   - 순위: {model_info['rank']}위")
        print(f"   - 목표 MAE: {model_info['mae']}")
        print(f"   - 목표 FPS: {model_info['fps']}")
        print(f"   - 총 파라미터 수: {total_params:,}")
        print(f"   - 훈련 가능 파라미터 수: {trainable_params:,}")
        print(f"   - 모델 구조: {self.model_type.value}")
        
        # 체크포인트 정보
        if 'epoch' in checkpoint:
            print(f"   - 훈련 에포크: {checkpoint['epoch']}")
        if 'val_mae' in checkpoint:
            print(f"   - 검증 MAE: {checkpoint['val_mae']:.4f}")
        if 'args' in checkpoint:
            print(f"   - 훈련 설정: {checkpoint['args']}")
    
    def predict(self, vision_features: torch.Tensor, text_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        추론 실행
        
        Args:
            vision_features: (batch_size, 2048) - Kosmos2 vision features
            text_features: (batch_size, 2048) - CLIP text features (하이브리드 모델에서만)
            
        Returns:
            actions: (batch_size, 2) - Predicted 2D actions
        """
        if self.model is None:
            raise ValueError("모델이 로드되지 않았습니다. load_model()을 먼저 호출하세요.")
        
        with torch.no_grad():
            # 디바이스로 이동
            vision_features = vision_features.to(self.device)
            if text_features is not None:
                text_features = text_features.to(self.device)
            
            # 추론
            actions = self.model(vision_features, text_features)
            
            return actions
    
    def get_model(self) -> Optional[UnifiedMobileVLAModel]:
        """로드된 모델 반환"""
        return self.model
    
    def get_model_info(self) -> Optional[Dict[str, Any]]:
        """현재 모델 정보 반환"""
        if self.model_type is None:
            return None
        return self.model_info[self.model_type]

def test_unified_loader():
    """통합 모델 로더 테스트"""
    print("🧪 통합 Mobile VLA 모델 로더 테스트")
    print("=" * 60)
    
    # 로더 생성
    loader = UnifiedMobileVLAModelLoader()
    
    # 사용 가능한 모델 목록 출력
    loader.list_available_models()
    
    # 각 모델 타입별로 테스트
    for model_type in ModelType:
        print(f"\n🎯 {model_type.value} 모델 테스트")
        print("-" * 40)
        
        # 모델 로드
        model = loader.load_model(model_type)
        
        if model is not None:
            print("✅ 모델 로딩 성공!")
            
            # 테스트 추론
            batch_size = 2
            vision_features = torch.randn(batch_size, 2048)
            
            if model_type == ModelType.KOSMOS2_CLIP_HYBRID:
                text_features = torch.randn(batch_size, 2048)
                actions = loader.predict(vision_features, text_features)
                print(f"✅ 하이브리드 모델 추론 성공: {actions.shape}")
            else:
                actions = loader.predict(vision_features)
                print(f"✅ 순수 Kosmos2 모델 추론 성공: {actions.shape}")
            
            # 성능 테스트
            print("\n⚡ 성능 테스트 중...")
            import time
            
            # CPU에서 테스트
            vision_features_cpu = torch.randn(1, 2048)
            if model_type == ModelType.KOSMOS2_CLIP_HYBRID:
                text_features_cpu = torch.randn(1, 2048)
            
            start_time = time.time()
            for _ in range(100):
                if model_type == ModelType.KOSMOS2_CLIP_HYBRID:
                    actions_cpu = loader.predict(vision_features_cpu, text_features_cpu)
                else:
                    actions_cpu = loader.predict(vision_features_cpu)
            cpu_time = time.time() - start_time
            
            print(f"   - CPU 추론 시간 (100회): {cpu_time:.4f}초")
            print(f"   - CPU 평균 추론 시간: {cpu_time/100*1000:.2f}ms")
            print(f"   - CPU FPS: {100/cpu_time:.1f}")
            
            # GPU에서 테스트 (CUDA 사용 가능한 경우)
            if torch.cuda.is_available():
                vision_features_gpu = torch.randn(1, 2048).cuda()
                if model_type == ModelType.KOSMOS2_CLIP_HYBRID:
                    text_features_gpu = torch.randn(1, 2048).cuda()
                
                # GPU 워밍업
                for _ in range(10):
                    if model_type == ModelType.KOSMOS2_CLIP_HYBRID:
                        actions_gpu = loader.predict(vision_features_gpu, text_features_gpu)
                    else:
                        actions_gpu = loader.predict(vision_features_gpu)
                
                torch.cuda.synchronize()
                start_time = time.time()
                for _ in range(100):
                    if model_type == ModelType.KOSMOS2_CLIP_HYBRID:
                        actions_gpu = loader.predict(vision_features_gpu, text_features_gpu)
                    else:
                        actions_gpu = loader.predict(vision_features_gpu)
                torch.cuda.synchronize()
                gpu_time = time.time() - start_time
                
                print(f"   - GPU 추론 시간 (100회): {gpu_time:.4f}초")
                print(f"   - GPU 평균 추론 시간: {gpu_time/100*1000:.2f}ms")
                print(f"   - GPU FPS: {100/gpu_time:.1f}")
                print(f"   - GPU 가속: {cpu_time/gpu_time:.1f}x")
        else:
            print("❌ 모델 로딩 실패")
    
    print("\n✅ 통합 테스트 완료!")

def main():
    """메인 함수"""
    test_unified_loader()

if __name__ == "__main__":
    main()
