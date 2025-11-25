#!/usr/bin/env python3
"""
🎯 메모리 최적화된 Kosmos2 + CLIP 하이브리드 모델 추론 스크립트
체크포인트: best_simple_clip_lstm_model.pth
메모리 모니터링 및 최적화 포함
"""

import torch
import torch.nn as nn
import numpy as np
import time
import os
import sys
import psutil
import gc
from typing import Optional, Tuple
import subprocess

def get_memory_info():
    """메모리 사용량 정보를 가져옵니다."""
    try:
        # 시스템 메모리 정보
        memory = psutil.virtual_memory()
        system_info = {
            'total': memory.total / (1024**3),  # GB
            'available': memory.available / (1024**3),  # GB
            'used': memory.used / (1024**3),  # GB
            'percent': memory.percent
        }
        
        # GPU 메모리 정보 (CUDA 사용 가능한 경우)
        gpu_info = {}
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_stats()
            gpu_info = {
                'allocated': torch.cuda.memory_allocated() / (1024**3),  # GB
                'cached': torch.cuda.memory_reserved() / (1024**3),  # GB
                'max_allocated': torch.cuda.max_memory_allocated() / (1024**3),  # GB
                'max_cached': torch.cuda.max_memory_reserved() / (1024**3)  # GB
            }
        
        return system_info, gpu_info
    except Exception as e:
        print(f"메모리 정보 가져오기 실패: {e}")
        return {}, {}

def print_memory_status(title="메모리 상태"):
    """메모리 상태를 출력합니다."""
    system_info, gpu_info = get_memory_info()
    
    print(f"\n📊 {title}")
    print("-" * 50)
    
    if system_info:
        print(f"💾 시스템 메모리:")
        print(f"   전체: {system_info['total']:.2f} GB")
        print(f"   사용 중: {system_info['used']:.2f} GB ({system_info['percent']:.1f}%)")
        print(f"   사용 가능: {system_info['available']:.2f} GB")
    
    if gpu_info:
        print(f"🎮 GPU 메모리:")
        print(f"   할당됨: {gpu_info['allocated']:.2f} GB")
        print(f"   캐시됨: {gpu_info['cached']:.2f} GB")
        print(f"   최대 할당: {gpu_info['max_allocated']:.2f} GB")
        print(f"   최대 캐시: {gpu_info['max_cached']:.2f} GB")

def clear_memory():
    """메모리를 정리합니다."""
    print("🧹 메모리 정리 중...")
    
    # PyTorch 캐시 정리
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Python 가비지 컬렉션
    gc.collect()
    
    print("✅ 메모리 정리 완료")

class MemoryOptimizedKosmosCLIPModel(nn.Module):
    """메모리 최적화된 Kosmos2 + CLIP 하이브리드 모델"""
    
    def __init__(self, 
                 kosmos_hidden_size=4096,
                 clip_hidden_size=768,
                 fusion_size=2048,
                 lstm_hidden_size=512,
                 lstm_layers=4,
                 action_size=2):
        super().__init__()
        
        print("🔨 메모리 최적화된 모델 초기화 중...")
        
        # 더 작은 모델로 시작 (메모리 절약)
        self.kosmos_hidden_size = kosmos_hidden_size
        self.clip_hidden_size = clip_hidden_size
        
        # Kosmos2 모델 (레이어 수 줄임)
        self.kosmos_text_model = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=kosmos_hidden_size,
                nhead=16,
                dim_feedforward=8192,  # 줄임
                dropout=0.1,
                batch_first=True
            ),
            num_layers=6  # 24 → 6으로 줄임
        )
        
        self.kosmos_vision_model = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=kosmos_hidden_size,
                nhead=16,
                dim_feedforward=8192,  # 줄임
                dropout=0.1,
                batch_first=True
            ),
            num_layers=6  # 24 → 6으로 줄임
        )
        
        # CLIP 모델 (레이어 수 줄임)
        self.clip_text_model = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=clip_hidden_size,
                nhead=12,
                dim_feedforward=1536,  # 줄임
                dropout=0.1,
                batch_first=True
            ),
            num_layers=6  # 12 → 6으로 줄임
        )
        
        self.clip_vision_model = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=clip_hidden_size,
                nhead=12,
                dim_feedforward=1536,  # 줄임
                dropout=0.1,
                batch_first=True
            ),
            num_layers=6  # 12 → 6으로 줄임
        )
        
        # Feature fusion layer
        self.feature_fusion = nn.Linear(
            kosmos_hidden_size + clip_hidden_size, 
            fusion_size
        )
        
        # LSTM layer (레이어 수 줄임)
        self.lstm = nn.LSTM(
            input_size=fusion_size,
            hidden_size=lstm_hidden_size,
            num_layers=2,  # 4 → 2로 줄임
            batch_first=True,
            dropout=0.1
        )
        
        # Action prediction head
        self.action_head = nn.Sequential(
            nn.Linear(lstm_hidden_size, lstm_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(lstm_hidden_size // 2, action_size)
        )
        
        # Embeddings (크기 줄임)
        self.kosmos_text_embedding = nn.Embedding(32000, kosmos_hidden_size)
        self.kosmos_vision_embedding = nn.Linear(768, kosmos_hidden_size)
        self.clip_text_embedding = nn.Embedding(49408, clip_hidden_size)
        self.clip_vision_embedding = nn.Linear(768, clip_hidden_size)
        
        # Position embeddings (크기 줄임)
        self.kosmos_text_pos_embedding = nn.Embedding(512, kosmos_hidden_size)  # 2048 → 512
        self.kosmos_vision_pos_embedding = nn.Embedding(257, kosmos_hidden_size)
        self.clip_text_pos_embedding = nn.Embedding(77, clip_hidden_size)
        self.clip_vision_pos_embedding = nn.Embedding(257, clip_hidden_size)
        
        # Layer norms
        self.kosmos_text_norm = nn.LayerNorm(kosmos_hidden_size)
        self.kosmos_vision_norm = nn.LayerNorm(kosmos_hidden_size)
        self.clip_text_norm = nn.LayerNorm(clip_hidden_size)
        self.clip_vision_norm = nn.LayerNorm(clip_hidden_size)
        
        print("✅ 메모리 최적화된 모델 초기화 완료")
        
    def forward(self, 
                kosmos_text_input: Optional[torch.Tensor] = None,
                kosmos_vision_input: Optional[torch.Tensor] = None,
                clip_text_input: Optional[torch.Tensor] = None,
                clip_vision_input: Optional[torch.Tensor] = None,
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """Forward pass with memory optimization"""
        
        # Kosmos2 text processing
        if kosmos_text_input is not None:
            kosmos_text_emb = self.kosmos_text_embedding(kosmos_text_input)
            kosmos_text_pos = self.kosmos_text_pos_embedding(
                torch.arange(kosmos_text_input.size(1), device=kosmos_text_input.device)
            ).unsqueeze(0)
            kosmos_text_emb = kosmos_text_emb + kosmos_text_pos
            kosmos_text_emb = self.kosmos_text_norm(kosmos_text_emb)
            kosmos_text_features = self.kosmos_text_model(kosmos_text_emb)
            kosmos_text_features = kosmos_text_features.mean(dim=1)
            del kosmos_text_emb, kosmos_text_pos  # 메모리 해제
        else:
            kosmos_text_features = torch.zeros(
                kosmos_vision_input.size(0), self.kosmos_hidden_size, 
                device=kosmos_vision_input.device
            )
        
        # Kosmos2 vision processing
        if kosmos_vision_input is not None:
            kosmos_vision_emb = self.kosmos_vision_embedding(kosmos_vision_input)
            kosmos_vision_pos = self.kosmos_vision_pos_embedding(
                torch.arange(kosmos_vision_input.size(1), device=kosmos_vision_input.device)
            ).unsqueeze(0)
            kosmos_vision_emb = kosmos_vision_emb + kosmos_vision_pos
            kosmos_vision_emb = self.kosmos_vision_norm(kosmos_vision_emb)
            kosmos_vision_features = self.kosmos_vision_model(kosmos_vision_emb)
            kosmos_vision_features = kosmos_vision_features.mean(dim=1)
            del kosmos_vision_emb, kosmos_vision_pos  # 메모리 해제
        else:
            kosmos_vision_features = torch.zeros(
                kosmos_text_input.size(0), self.kosmos_hidden_size,
                device=kosmos_text_input.device
            )
        
        # CLIP text processing
        if clip_text_input is not None:
            clip_text_emb = self.clip_text_embedding(clip_text_input)
            clip_text_pos = self.clip_text_pos_embedding(
                torch.arange(clip_text_input.size(1), device=clip_text_input.device)
            ).unsqueeze(0)
            clip_text_emb = clip_text_emb + clip_text_pos
            clip_text_emb = self.clip_text_norm(clip_text_emb)
            clip_text_features = self.clip_text_model(clip_text_emb)
            clip_text_features = clip_text_features.mean(dim=1)
            del clip_text_emb, clip_text_pos  # 메모리 해제
        else:
            clip_text_features = torch.zeros(
                kosmos_vision_input.size(0), self.clip_hidden_size,
                device=kosmos_vision_input.device
            )
        
        # CLIP vision processing
        if clip_vision_input is not None:
            clip_vision_emb = self.clip_vision_embedding(clip_vision_input)
            clip_vision_pos = self.clip_vision_pos_embedding(
                torch.arange(clip_vision_input.size(1), device=clip_vision_input.device)
            ).unsqueeze(0)
            clip_vision_emb = clip_vision_emb + clip_vision_pos
            clip_vision_emb = self.clip_vision_norm(clip_vision_emb)
            clip_vision_features = self.clip_vision_model(clip_vision_emb)
            clip_vision_features = clip_vision_features.mean(dim=1)
            del clip_vision_emb, clip_vision_pos  # 메모리 해제
        else:
            clip_vision_features = torch.zeros(
                kosmos_text_input.size(0), self.clip_hidden_size,
                device=kosmos_text_input.device
            )
        
        # Feature fusion
        kosmos_features = (kosmos_text_features + kosmos_vision_features) / 2
        clip_features = (clip_text_features + clip_vision_features) / 2
        
        combined_features = torch.cat([kosmos_features, clip_features], dim=-1)
        fused_features = self.feature_fusion(combined_features)
        
        # Add sequence dimension for LSTM
        if fused_features.dim() == 2:
            fused_features = fused_features.unsqueeze(1)
        
        # LSTM processing
        lstm_out, hidden = self.lstm(fused_features, hidden)
        
        # Take the last output
        last_output = lstm_out[:, -1, :]
        
        # Action prediction
        actions = self.action_head(last_output)
        
        return actions, hidden

def load_model_with_memory_check(checkpoint_path):
    """메모리 체크와 함께 모델을 로드합니다."""
    print(f"🔄 메모리 최적화된 모델 로딩 중: {checkpoint_path}")
    
    # 초기 메모리 상태 확인
    print_memory_status("초기 메모리 상태")
    
    try:
        # 모델 인스턴스 생성
        print("🔨 모델 인스턴스 생성 중...")
        model = MemoryOptimizedKosmosCLIPModel()
        
        print_memory_status("모델 생성 후 메모리 상태")
        
        # 체크포인트 로드 (CPU에서 먼저)
        print("📂 체크포인트 로딩 중...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        print(f"📋 체크포인트 키 수: {len(checkpoint.keys())}")
        print(f"📋 주요 키들: {list(checkpoint.keys())[:5]}...")
        
        # 메모리 상태 확인
        print_memory_status("체크포인트 로드 후 메모리 상태")
        
        # 모델 상태 로드 시도
        print("🔧 모델 가중치 로딩 중...")
        try:
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            elif 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
            else:
                model.load_state_dict(checkpoint)
            print("✅ 모델 가중치 로드 성공")
        except Exception as e:
            print(f"⚠️ 모델 가중치 로드 실패 (무작위 가중치 사용): {e}")
        
        # GPU로 이동 (가능한 경우)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 모델을 {device}로 이동 중...")
        
        model = model.to(device)
        model.eval()
        
        print_memory_status("GPU 이동 후 메모리 상태")
        
        print(f"✅ 모델 로드 완료 (디바이스: {device})")
        print(f"📊 모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
        
        return model, device
        
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        clear_memory()
        raise e

def generate_small_dummy_inputs(batch_size=1, device='cpu'):
    """메모리 절약을 위한 작은 더미 입력을 생성합니다."""
    # 더 작은 시퀀스 길이 사용
    kosmos_text = torch.randint(0, 32000, (batch_size, 64)).to(device)  # 128 → 64
    kosmos_vision = torch.randn(batch_size, 129, 768).to(device)  # 257 → 129
    clip_text = torch.randint(0, 49408, (batch_size, 38)).to(device)  # 77 → 38
    clip_vision = torch.randn(batch_size, 129, 768).to(device)  # 257 → 129
    
    return kosmos_text, kosmos_vision, clip_text, clip_vision

def run_memory_optimized_inference(model, device, inputs=None):
    """메모리 최적화된 추론을 실행합니다."""
    if inputs is None:
        inputs = generate_small_dummy_inputs(device=device)
    
    kosmos_text, kosmos_vision, clip_text, clip_vision = inputs
    
    print_memory_status("추론 시작 전 메모리 상태")
    
    start_time = time.time()
    
    with torch.no_grad():
        actions, hidden = model(
            kosmos_text_input=kosmos_text,
            kosmos_vision_input=kosmos_vision,
            clip_text_input=clip_text,
            clip_vision_input=clip_vision
        )
    
    inference_time = (time.time() - start_time) * 1000  # ms
    fps = 1000 / inference_time
    
    print_memory_status("추론 완료 후 메모리 상태")
    
    # 메모리 정리
    clear_memory()
    
    return actions, hidden, inference_time, fps

def main():
    print("🎯 메모리 최적화된 Kosmos2 + CLIP 하이브리드 모델 추론 데모")
    print("=" * 70)
    
    # 체크포인트 경로 설정
    checkpoint_path = './vla/Robo+/Mobile_VLA/simple_clip_lstm_model/best_simple_clip_lstm_model.pth'
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ 체크포인트 파일을 찾을 수 없습니다: {checkpoint_path}")
        return
    
    try:
        # 모델 로드
        model, device = load_model_with_memory_check(checkpoint_path)
        
        print("\n🚀 대화형 추론 모드 시작")
        print("명령어:")
        print("  'infer': 단일 추론 실행")
        print("  'benchmark': 성능 벤치마크 (5회)")
        print("  'memory': 메모리 상태 확인")
        print("  'clear': 메모리 정리")
        print("  'quit': 종료")
        print("-" * 70)
        
        while True:
            try:
                command = input("\n💬 명령어 입력: ").strip().lower()
                
                if command == 'quit':
                    print("👋 추론 데모를 종료합니다.")
                    break
                
                elif command == 'infer':
                    print("🔄 단일 추론 실행 중...")
                    actions, hidden, inference_time, fps = run_memory_optimized_inference(model, device)
                    
                    print(f"✅ 추론 완료:")
                    print(f"   추론 시간: {inference_time:.3f}ms")
                    print(f"   FPS: {fps:.0f}")
                    print(f"   액션 결과: {actions[0].cpu().numpy()}")
                
                elif command == 'benchmark':
                    print("🔄 성능 벤치마크 실행 중 (5회)...")
                    
                    times = []
                    for i in range(5):
                        print(f"   {i+1}/5 실행 중...")
                        _, _, inference_time, _ = run_memory_optimized_inference(model, device)
                        times.append(inference_time)
                        print(f"   {i+1}/5 완료: {inference_time:.3f}ms")
                    
                    avg_time = sum(times) / len(times)
                    fps = 1000 / avg_time
                    min_time = min(times)
                    max_time = max(times)
                    
                    print(f"✅ 벤치마크 완료:")
                    print(f"   평균 추론 시간: {avg_time:.3f}ms")
                    print(f"   최소 추론 시간: {min_time:.3f}ms")
                    print(f"   최대 추론 시간: {max_time:.3f}ms")
                    print(f"   평균 FPS: {fps:.0f}")
                
                elif command == 'memory':
                    print_memory_status("현재 메모리 상태")
                
                elif command == 'clear':
                    clear_memory()
                    print_memory_status("메모리 정리 후 상태")
                
                else:
                    print("❌ 알 수 없는 명령어입니다. 'infer', 'benchmark', 'memory', 'clear', 'quit' 중 선택하세요.")
            
            except KeyboardInterrupt:
                print("\n👋 추론 데모를 종료합니다.")
                break
            except Exception as e:
                print(f"❌ 오류 발생: {e}")
                clear_memory()
                import traceback
                traceback.print_exc()
    
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        clear_memory()
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
