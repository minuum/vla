#!/usr/bin/env python3
"""
로컬 Mobile VLA 추론 테스트 스크립트
Jetson 환경에서 CUDA 지원 PyTorch를 사용한 추론 테스트
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import time
import os
import sys
from typing import Optional, Tuple

# CUDA 테스트
def test_cuda():
    """CUDA 사용 가능 여부 테스트"""
    print("=" * 60)
    print("🚀 CUDA 테스트")
    print("=" * 60)
    
    print(f"📦 PyTorch 버전: {torch.__version__}")
    print(f"🔧 CUDA 사용 가능: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"📱 CUDA 디바이스 수: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            device_name = torch.cuda.get_device_name(i)
            device_capability = torch.cuda.get_device_capability(i)
            print(f"   디바이스 {i}: {device_name} (Compute Capability: {device_capability})")
        
        # 메모리 정보
        memory_total = torch.cuda.get_device_properties(0).total_memory
        print(f"💾 총 메모리: {memory_total / 1024**2:.1f} MB")
        
        # 간단한 CUDA 연산 테스트
        try:
            print("\n🧪 CUDA 연산 테스트...")
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = torch.mm(x, y)
            print("✅ CUDA 연산 성공!")
            print(f"   결과 텐서 크기: {z.shape}")
            print(f"   결과 텐서 디바이스: {z.device}")
            return True
        except Exception as e:
            print(f"❌ CUDA 연산 실패: {e}")
            return False
    else:
        print("❌ CUDA를 사용할 수 없습니다.")
        return False

class SimpleMobileVLAModel(nn.Module):
    """간단한 Mobile VLA 모델 (테스트용)"""
    
    def __init__(self, input_dim: int = 2048, hidden_dim: int = 1024, action_dim: int = 2):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        
        # 간단한 MLP 구조
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """순전파"""
        return self.encoder(x)

def test_model_inference():
    """모델 추론 테스트"""
    print("\n" + "=" * 60)
    print("🧠 모델 추론 테스트")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 사용 디바이스: {device}")
    
    # 모델 생성
    model = SimpleMobileVLAModel().to(device)
    model.eval()
    
    print(f"📊 모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    
    # 테스트 데이터 생성
    batch_size = 1
    input_dim = 2048
    test_input = torch.randn(batch_size, input_dim).to(device)
    
    print(f"📥 입력 크기: {test_input.shape}")
    
    # 추론 시간 측정
    num_runs = 100
    times = []
    
    # 워밍업
    with torch.no_grad():
        for _ in range(10):
            _ = model(test_input)
    
    # 실제 측정
    with torch.no_grad():
        for i in range(num_runs):
            start_time = time.time()
            output = model(test_input)
            end_time = time.time()
            times.append(end_time - start_time)
    
    avg_time = np.mean(times)
    fps = 1.0 / avg_time
    
    print(f"📤 출력 크기: {output.shape}")
    print(f"⏱️ 평균 추론 시간: {avg_time*1000:.2f} ms")
    print(f"🚀 추론 FPS: {fps:.1f}")
    
    return True

def test_image_processing():
    """이미지 처리 테스트"""
    print("\n" + "=" * 60)
    print("📷 이미지 처리 테스트")
    print("=" * 60)
    
    # 테스트 이미지 생성
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    print(f"📐 원본 이미지 크기: {test_image.shape}")
    
    # OpenCV 처리
    start_time = time.time()
    resized = cv2.resize(test_image, (224, 224))
    normalized = resized.astype(np.float32) / 255.0
    cv_time = time.time() - start_time
    
    print(f"🔧 OpenCV 처리 시간: {cv_time*1000:.2f} ms")
    print(f"📐 리사이즈된 이미지 크기: {resized.shape}")
    
    # PIL 처리
    pil_image = Image.fromarray(test_image)
    start_time = time.time()
    pil_resized = pil_image.resize((224, 224))
    pil_time = time.time() - start_time
    
    print(f"🖼️ PIL 처리 시간: {pil_time*1000:.2f} ms")
    
    return True

def test_transformers():
    """Transformers 라이브러리 테스트"""
    print("\n" + "=" * 60)
    print("🤗 Transformers 테스트")
    print("=" * 60)
    
    try:
        from transformers import AutoProcessor, AutoModel
        print("✅ Transformers 라이브러리 사용 가능")
        
        # 간단한 모델 로드 테스트 (실제 다운로드는 하지 않음)
        print("🔄 모델 정보 확인 중...")
        
        # 사용 가능한 모델 목록
        available_models = [
            "microsoft/kosmos-2-patch14-224",
            "openai/clip-vit-base-patch32",
            "minium/mobile-vla"
        ]
        
        print("📋 사용 가능한 모델:")
        for model_name in available_models:
            print(f"   - {model_name}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Transformers 라이브러리 없음: {e}")
        return False
    except Exception as e:
        print(f"❌ Transformers 테스트 실패: {e}")
        return False

def main():
    """메인 함수"""
    print("🚀 Mobile VLA 로컬 추론 환경 테스트")
    print("=" * 60)
    
    # 1. CUDA 테스트
    cuda_ok = test_cuda()
    
    # 2. 모델 추론 테스트
    model_ok = test_model_inference()
    
    # 3. 이미지 처리 테스트
    image_ok = test_image_processing()
    
    # 4. Transformers 테스트
    transformers_ok = test_transformers()
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("📊 테스트 결과 요약")
    print("=" * 60)
    
    results = {
        "CUDA 지원": "✅" if cuda_ok else "❌",
        "모델 추론": "✅" if model_ok else "❌",
        "이미지 처리": "✅" if image_ok else "❌",
        "Transformers": "✅" if transformers_ok else "❌"
    }
    
    for test_name, result in results.items():
        print(f"{test_name}: {result}")
    
    all_passed = all([cuda_ok, model_ok, image_ok, transformers_ok])
    
    if all_passed:
        print("\n🎉 모든 테스트 통과! 로컬 추론 환경이 준비되었습니다.")
        print("\n📋 다음 단계:")
        print("   1. 실제 Mobile VLA 모델 로드")
        print("   2. 카메라 입력 처리")
        print("   3. 실시간 추론 파이프라인 구성")
    else:
        print("\n⚠️ 일부 테스트 실패. 환경 설정을 확인해주세요.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
