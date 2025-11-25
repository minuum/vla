#!/usr/bin/env python3
"""
🎯 Kosmos2 + CLIP 하이브리드 모델 추론 스크립트
체크포인트: best_simple_clip_lstm_model.pth
실제 모델 구조: Kosmos2 (24층) + CLIP (12층) + LSTM + Action Head
"""

import torch
import torch.nn as nn
import numpy as np
import time
import os
import sys
from typing import Optional, Tuple

class KosmosCLIPHybridModel(nn.Module):
    """Kosmos2 + CLIP 하이브리드 모델"""
    
    def __init__(self, 
                 kosmos_hidden_size=4096,
                 clip_hidden_size=768,
                 fusion_size=2048,
                 lstm_hidden_size=512,
                 lstm_layers=4,
                 action_size=2):
        super().__init__()
        
        # Kosmos2 모델 (24층 Transformer)
        self.kosmos_text_model = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=kosmos_hidden_size,
                nhead=16,
                dim_feedforward=16384,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=24
        )
        
        self.kosmos_vision_model = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=kosmos_hidden_size,
                nhead=16,
                dim_feedforward=16384,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=24
        )
        
        # CLIP 모델 (12층 Transformer)
        self.clip_text_model = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=clip_hidden_size,
                nhead=12,
                dim_feedforward=3072,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=12
        )
        
        self.clip_vision_model = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=clip_hidden_size,
                nhead=12,
                dim_feedforward=3072,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=12
        )
        
        # Feature fusion layer
        self.feature_fusion = nn.Linear(
            kosmos_hidden_size + clip_hidden_size, 
            fusion_size
        )
        
        # LSTM layer (4층)
        self.lstm = nn.LSTM(
            input_size=fusion_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.1
        )
        
        # Action prediction head
        self.action_head = nn.Sequential(
            nn.Linear(lstm_hidden_size, lstm_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(lstm_hidden_size // 2, lstm_hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(lstm_hidden_size // 4, action_size)
        )
        
        # Embeddings
        self.kosmos_text_embedding = nn.Embedding(32000, kosmos_hidden_size)
        self.kosmos_vision_embedding = nn.Linear(768, kosmos_hidden_size)  # ViT patch size
        self.clip_text_embedding = nn.Embedding(49408, clip_hidden_size)
        self.clip_vision_embedding = nn.Linear(768, clip_hidden_size)
        
        # Position embeddings
        self.kosmos_text_pos_embedding = nn.Embedding(2048, kosmos_hidden_size)
        self.kosmos_vision_pos_embedding = nn.Embedding(257, kosmos_hidden_size)  # 16x16 + 1
        self.clip_text_pos_embedding = nn.Embedding(77, clip_hidden_size)
        self.clip_vision_pos_embedding = nn.Embedding(257, clip_hidden_size)
        
        # Layer norms
        self.kosmos_text_norm = nn.LayerNorm(kosmos_hidden_size)
        self.kosmos_vision_norm = nn.LayerNorm(kosmos_hidden_size)
        self.clip_text_norm = nn.LayerNorm(clip_hidden_size)
        self.clip_vision_norm = nn.LayerNorm(clip_hidden_size)
        
    def forward(self, 
                kosmos_text_input: Optional[torch.Tensor] = None,
                kosmos_vision_input: Optional[torch.Tensor] = None,
                clip_text_input: Optional[torch.Tensor] = None,
                clip_vision_input: Optional[torch.Tensor] = None,
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """
        Forward pass
        
        Args:
            kosmos_text_input: Kosmos2 text input [batch, seq_len]
            kosmos_vision_input: Kosmos2 vision input [batch, seq_len, 768]
            clip_text_input: CLIP text input [batch, seq_len]
            clip_vision_input: CLIP vision input [batch, seq_len, 768]
            hidden: LSTM hidden state
        """
        
        # Kosmos2 text processing
        if kosmos_text_input is not None:
            kosmos_text_emb = self.kosmos_text_embedding(kosmos_text_input)
            kosmos_text_pos = self.kosmos_text_pos_embedding(
                torch.arange(kosmos_text_input.size(1), device=kosmos_text_input.device)
            ).unsqueeze(0)
            kosmos_text_emb = kosmos_text_emb + kosmos_text_pos
            kosmos_text_emb = self.kosmos_text_norm(kosmos_text_emb)
            kosmos_text_features = self.kosmos_text_model(kosmos_text_emb)
            kosmos_text_features = kosmos_text_features.mean(dim=1)  # Global pooling
        else:
            kosmos_text_features = torch.zeros(
                kosmos_vision_input.size(0), 4096, 
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
            kosmos_vision_features = kosmos_vision_features.mean(dim=1)  # Global pooling
        else:
            kosmos_vision_features = torch.zeros(
                kosmos_text_input.size(0), 4096,
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
            clip_text_features = clip_text_features.mean(dim=1)  # Global pooling
        else:
            clip_text_features = torch.zeros(
                kosmos_vision_input.size(0), 768,
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
            clip_vision_features = clip_vision_features.mean(dim=1)  # Global pooling
        else:
            clip_vision_features = torch.zeros(
                kosmos_text_input.size(0), 768,
                device=kosmos_text_input.device
            )
        
        # Feature fusion
        kosmos_features = (kosmos_text_features + kosmos_vision_features) / 2
        clip_features = (clip_text_features + clip_vision_features) / 2
        
        combined_features = torch.cat([kosmos_features, clip_features], dim=-1)
        fused_features = self.feature_fusion(combined_features)
        
        # Add sequence dimension for LSTM
        if fused_features.dim() == 2:
            fused_features = fused_features.unsqueeze(1)  # [batch, 1, features]
        
        # LSTM processing
        lstm_out, hidden = self.lstm(fused_features, hidden)
        
        # Take the last output
        last_output = lstm_out[:, -1, :]
        
        # Action prediction
        actions = self.action_head(last_output)
        
        return actions, hidden

def load_model(checkpoint_path):
    """모델과 체크포인트를 로드합니다."""
    print(f"🔄 Kosmos2 + CLIP 하이브리드 모델 로딩 중: {checkpoint_path}")
    
    # 모델 인스턴스 생성
    model = KosmosCLIPHybridModel()
    
    # 체크포인트 로드
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 체크포인트 구조 확인
    print(f"📋 체크포인트 키 수: {len(checkpoint.keys())}")
    print(f"📋 주요 키들: {list(checkpoint.keys())[:10]}...")
    
    # 모델 상태 로드 (다양한 키 이름 시도)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        # 직접 로드 시도
        model.load_state_dict(checkpoint)
    
    # GPU로 이동 (가능한 경우)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    print(f"✅ 모델 로드 완료 (디바이스: {device})")
    print(f"📊 모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, device

def generate_dummy_inputs(batch_size=1, device='cpu'):
    """더미 입력을 생성합니다."""
    # Kosmos2 inputs
    kosmos_text = torch.randint(0, 32000, (batch_size, 128)).to(device)  # 128 tokens
    kosmos_vision = torch.randn(batch_size, 257, 768).to(device)  # 16x16 + 1 patches
    
    # CLIP inputs
    clip_text = torch.randint(0, 49408, (batch_size, 77)).to(device)  # 77 tokens
    clip_vision = torch.randn(batch_size, 257, 768).to(device)  # 16x16 + 1 patches
    
    return kosmos_text, kosmos_vision, clip_text, clip_vision

def run_inference(model, device, inputs=None):
    """추론을 실행합니다."""
    if inputs is None:
        inputs = generate_dummy_inputs(device=device)
    
    kosmos_text, kosmos_vision, clip_text, clip_vision = inputs
    
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
    
    return actions, hidden, inference_time, fps

def main():
    print("🎯 Kosmos2 + CLIP 하이브리드 모델 추론 데모")
    print("=" * 60)
    
    # 체크포인트 경로 설정
    checkpoint_path = './vla/Robo+/Mobile_VLA/simple_clip_lstm_model/best_simple_clip_lstm_model.pth'
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ 체크포인트 파일을 찾을 수 없습니다: {checkpoint_path}")
        return
    
    try:
        # 모델 로드
        model, device = load_model(checkpoint_path)
        
        print("\n🚀 대화형 추론 모드 시작")
        print("명령어:")
        print("  'infer': 단일 추론 실행")
        print("  'benchmark': 성능 벤치마크 (10회)")
        print("  'continuous': 연속 추론 모드")
        print("  'quit': 종료")
        print("-" * 60)
        
        while True:
            try:
                command = input("\n💬 명령어 입력: ").strip().lower()
                
                if command == 'quit':
                    print("👋 추론 데모를 종료합니다.")
                    break
                
                elif command == 'infer':
                    print("🔄 단일 추론 실행 중...")
                    actions, hidden, inference_time, fps = run_inference(model, device)
                    
                    print(f"✅ 추론 완료:")
                    print(f"   추론 시간: {inference_time:.3f}ms")
                    print(f"   FPS: {fps:.0f}")
                    print(f"   액션 결과: {actions[0].cpu().numpy()}")
                
                elif command == 'benchmark':
                    print("🔄 성능 벤치마크 실행 중 (10회)...")
                    
                    times = []
                    for i in range(10):
                        _, _, inference_time, _ = run_inference(model, device)
                        times.append(inference_time)
                        print(f"   {i+1}/10 완료: {inference_time:.3f}ms")
                    
                    avg_time = sum(times) / len(times)
                    fps = 1000 / avg_time
                    min_time = min(times)
                    max_time = max(times)
                    
                    print(f"✅ 벤치마크 완료:")
                    print(f"   평균 추론 시간: {avg_time:.3f}ms")
                    print(f"   최소 추론 시간: {min_time:.3f}ms")
                    print(f"   최대 추론 시간: {max_time:.3f}ms")
                    print(f"   평균 FPS: {fps:.0f}")
                
                elif command == 'continuous':
                    print("🔄 연속 추론 모드 시작 (Ctrl+C로 중단)")
                    print("실시간 FPS 모니터링...")
                    
                    try:
                        count = 0
                        start_time = time.time()
                        
                        while True:
                            _, _, inference_time, fps = run_inference(model, device)
                            count += 1
                            
                            if count % 5 == 0:
                                elapsed = time.time() - start_time
                                avg_fps = count / elapsed
                                print(f"   {count}회 완료 - 현재 FPS: {fps:.0f}, 평균 FPS: {avg_fps:.0f}")
                    
                    except KeyboardInterrupt:
                        elapsed = time.time() - start_time
                        avg_fps = count / elapsed
                        print(f"\n⏹️ 연속 추론 중단")
                        print(f"   총 추론 횟수: {count}")
                        print(f"   총 소요 시간: {elapsed:.2f}초")
                        print(f"   평균 FPS: {avg_fps:.0f}")
                
                else:
                    print("❌ 알 수 없는 명령어입니다. 'infer', 'benchmark', 'continuous', 'quit' 중 선택하세요.")
            
            except KeyboardInterrupt:
                print("\n👋 추론 데모를 종료합니다.")
                break
            except Exception as e:
                print(f"❌ 오류 발생: {e}")
                import traceback
                traceback.print_exc()
    
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
