#!/usr/bin/env python3
"""
🎯 Simple CLIP LSTM 모델 대화형 추론 데모
체크포인트: best_simple_clip_lstm_model.pth
"""

import torch
import torch.nn as nn
import numpy as np
import time
import os
import sys

# Simple CLIP LSTM 모델 클래스 정의
class SimpleCLIPLSTMModel(nn.Module):
    def __init__(self, input_size=2048, hidden_size=512, num_layers=2, output_size=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Vision feature encoder
        self.vision_encoder = nn.Linear(input_size, hidden_size)
        
        # Text feature encoder  
        self.text_encoder = nn.Linear(input_size, hidden_size)
        
        # LSTM layer
        self.lstm = nn.LSTM(hidden_size * 2, hidden_size, num_layers, batch_first=True)
        
        # Action prediction head
        self.action_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, output_size)
        )
        
    def forward(self, vision_features, text_features, hidden=None):
        # Encode features
        vision_encoded = self.vision_encoder(vision_features)
        text_encoded = self.text_encoder(text_features)
        
        # Concatenate features
        combined = torch.cat([vision_encoded, text_encoded], dim=-1)
        
        # Add sequence dimension if needed
        if combined.dim() == 2:
            combined = combined.unsqueeze(1)  # [batch, 1, features]
        
        # LSTM forward pass
        lstm_out, hidden = self.lstm(combined, hidden)
        
        # Take the last output
        last_output = lstm_out[:, -1, :]
        
        # Predict actions
        actions = self.action_head(last_output)
        
        return actions, hidden

def load_model(checkpoint_path):
    """모델과 체크포인트를 로드합니다."""
    print(f"🔄 모델 로딩 중: {checkpoint_path}")
    
    # 모델 인스턴스 생성
    model = SimpleCLIPLSTMModel()
    
    # 체크포인트 로드
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 체크포인트 구조 확인
    print(f"📋 체크포인트 키: {list(checkpoint.keys())}")
    
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

def generate_dummy_features(batch_size=1, device='cpu'):
    """더미 vision과 text features를 생성합니다."""
    vision_features = torch.randn(batch_size, 2048).to(device)
    text_features = torch.randn(batch_size, 2048).to(device)
    return vision_features, text_features

def run_inference(model, device, vision_features=None, text_features=None):
    """추론을 실행합니다."""
    if vision_features is None or text_features is None:
        vision_features, text_features = generate_dummy_features(device=device)
    
    start_time = time.time()
    
    with torch.no_grad():
        actions, hidden = model(vision_features, text_features)
    
    inference_time = (time.time() - start_time) * 1000  # ms
    fps = 1000 / inference_time
    
    return actions, hidden, inference_time, fps

def main():
    print("🎯 Simple CLIP LSTM 모델 대화형 추론 데모")
    print("=" * 50)
    
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
        print("  'benchmark': 성능 벤치마크 (100회)")
        print("  'continuous': 연속 추론 모드")
        print("  'quit': 종료")
        print("-" * 50)
        
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
                    print("🔄 성능 벤치마크 실행 중 (100회)...")
                    
                    times = []
                    for i in range(100):
                        _, _, inference_time, _ = run_inference(model, device)
                        times.append(inference_time)
                        
                        if (i + 1) % 20 == 0:
                            print(f"   진행률: {i + 1}/100")
                    
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
                            
                            if count % 10 == 0:
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
    
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
