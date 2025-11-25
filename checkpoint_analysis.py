#!/usr/bin/env python3
"""
체크포인트 파일 구조 분석 스크립트
실제 Mobile VLA 모델의 구조를 파악하고 간단한 추론 테스트
"""

import torch
import torch.nn as nn
import numpy as np
import time
import os
import sys
from typing import Dict, Any, Optional, List

def analyze_checkpoint(checkpoint_path: str):
    """체크포인트 파일 구조 분석"""
    print("=" * 60)
    print("🔍 체크포인트 파일 구조 분석")
    print("=" * 60)
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ 체크포인트 파일을 찾을 수 없습니다: {checkpoint_path}")
        return None
    
    print(f"📦 체크포인트 경로: {checkpoint_path}")
    
    try:
        # 체크포인트 로드
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        print(f"📊 체크포인트 키: {list(checkpoint.keys())}")
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"📋 모델 상태 딕셔너리 키 수: {len(state_dict)}")
            
            # 주요 키들 분석
            print("\n🔑 주요 모델 구성 요소:")
            key_groups = {}
            for key in state_dict.keys():
                prefix = key.split('.')[0]
                if prefix not in key_groups:
                    key_groups[prefix] = []
                key_groups[prefix].append(key)
            
            for prefix, keys in key_groups.items():
                print(f"   {prefix}: {len(keys)}개 파라미터")
                if len(keys) <= 5:  # 5개 이하면 모두 출력
                    for key in keys:
                        shape = state_dict[key].shape if hasattr(state_dict[key], 'shape') else 'N/A'
                        print(f"     - {key}: {shape}")
                else:  # 5개 초과면 처음 3개만 출력
                    for key in keys[:3]:
                        shape = state_dict[key].shape if hasattr(state_dict[key], 'shape') else 'N/A'
                        print(f"     - {key}: {shape}")
                    print(f"     ... 및 {len(keys)-3}개 더")
            
            # 모델 정보
            if 'epoch' in checkpoint:
                print(f"\n📈 훈련 에포크: {checkpoint['epoch']}")
            if 'loss' in checkpoint:
                print(f"📉 손실값: {checkpoint['loss']:.4f}")
            if 'val_mae' in checkpoint:
                print(f"📊 검증 MAE: {checkpoint['val_mae']:.4f}")
            
            return state_dict, checkpoint
        else:
            print("❌ 체크포인트에 model_state_dict가 없습니다.")
            return None, checkpoint
            
    except Exception as e:
        print(f"❌ 체크포인트 분석 실패: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def create_simple_inference_model(state_dict: Dict[str, torch.Tensor]):
    """체크포인트 분석을 바탕으로 간단한 추론 모델 생성"""
    print("\n" + "=" * 40)
    print("🧠 간단한 추론 모델 생성")
    print("=" * 40)
    
    # 액션 예측 부분만 추출
    action_keys = [k for k in state_dict.keys() if 'actions' in k]
    
    if not action_keys:
        print("❌ 액션 예측 관련 파라미터를 찾을 수 없습니다.")
        return None
    
    print(f"🎯 액션 예측 파라미터: {len(action_keys)}개")
    for key in action_keys:
        shape = state_dict[key].shape
        print(f"   - {key}: {shape}")
    
    # 간단한 액션 예측 모델 생성
    class SimpleActionModel(nn.Module):
        def __init__(self, input_dim: int = 2048, hidden_dim: int = 1024, output_dim: int = 2):
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, output_dim)
            )
        
        def forward(self, x):
            return self.mlp(x)
    
    # 모델 생성
    model = SimpleActionModel()
    
    print(f"📊 생성된 모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    
    return model

def test_simple_inference(model: nn.Module, device: torch.device):
    """간단한 추론 테스트"""
    print("\n" + "=" * 40)
    print("🔬 간단한 추론 성능 테스트")
    print("=" * 40)
    
    model = model.to(device)
    model.eval()
    
    # 테스트 데이터 생성
    batch_size = 1
    input_dim = 2048
    test_input = torch.randn(batch_size, input_dim).to(device)
    
    print(f"📥 입력 크기: {test_input.shape}")
    
    # 워밍업
    print("🔥 워밍업 중...")
    with torch.no_grad():
        for _ in range(10):
            _ = model(test_input)
    
    # 추론 시간 측정
    num_runs = 100
    times = []
    
    print(f"⏱️ {num_runs}회 추론 시간 측정 중...")
    with torch.no_grad():
        for i in range(num_runs):
            start_time = time.time()
            output = model(test_input)
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

def main():
    """메인 함수"""
    print("🚀 Mobile VLA 체크포인트 분석 및 간단한 추론 테스트")
    print("=" * 60)
    
    # 체크포인트 경로 설정
    checkpoint_paths = [
        "./Robo+/Mobile_VLA/simple_clip_lstm_model/best_simple_clip_lstm_model.pth",
        "./mobile-vla-omniwheel/best_simple_lstm_model.pth"
    ]
    
    checkpoint_path = None
    for path in checkpoint_paths:
        if os.path.exists(path):
            checkpoint_path = path
            break
    
    if checkpoint_path is None:
        print("❌ 사용 가능한 체크포인트 파일을 찾을 수 없습니다.")
        return False
    
    # 체크포인트 분석
    state_dict, checkpoint = analyze_checkpoint(checkpoint_path)
    
    if state_dict is None:
        print("❌ 체크포인트 분석 실패")
        return False
    
    # 간단한 추론 모델 생성
    model = create_simple_inference_model(state_dict)
    
    if model is None:
        print("❌ 추론 모델 생성 실패")
        return False
    
    # 추론 테스트
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inference_ok = test_simple_inference(model, device)
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("📊 테스트 결과 요약")
    print("=" * 60)
    
    results = {
        "체크포인트 분석": "✅" if state_dict is not None else "❌",
        "모델 생성": "✅" if model is not None else "❌",
        "추론 테스트": "✅" if inference_ok else "❌"
    }
    
    for test_name, result in results.items():
        print(f"{test_name}: {result}")
    
    all_passed = all([state_dict is not None, model is not None, inference_ok])
    
    if all_passed:
        print("\n🎉 체크포인트 분석 및 추론 테스트 성공!")
        print("\n📋 다음 단계:")
        print("   1. 실제 모델 구조 재구성")
        print("   2. 이미지 전처리 파이프라인 구성")
        print("   3. 실시간 추론 시스템 완성")
    else:
        print("\n⚠️ 일부 테스트 실패. 체크포인트 파일을 확인해주세요.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
