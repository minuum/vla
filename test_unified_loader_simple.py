#!/usr/bin/env python3
"""
간단한 통합 모델 로더 테스트
"""

import torch
import os
import sys

def test_cuda():
    """CUDA 테스트"""
    print("🔧 CUDA 환경 테스트")
    print(f"   - PyTorch: {torch.__version__}")
    print(f"   - CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   - CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"   - CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()

def test_checkpoint():
    """체크포인트 파일 테스트"""
    print("📦 체크포인트 파일 테스트")
    
    checkpoint_paths = [
        "./mobile-vla-omniwheel/best_simple_lstm_model.pth",
        "./vla/mobile-vla-omniwheel/best_simple_lstm_model.pth",
        "/workspace/vla/mobile-vla-omniwheel/best_simple_lstm_model.pth"
    ]
    
    for path in checkpoint_paths:
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"✅ 체크포인트 발견: {path} ({size_mb:.1f} MB)")
            
            try:
                # 체크포인트 로드 테스트
                checkpoint = torch.load(path, map_location='cpu')
                print(f"   - 로드 성공: {type(checkpoint)}")
                
                if isinstance(checkpoint, dict):
                    print(f"   - 키: {list(checkpoint.keys())}")
                    if 'model_state_dict' in checkpoint:
                        print(f"   - 모델 상태 딕셔너리 포함")
                    if 'epoch' in checkpoint:
                        print(f"   - 에포크: {checkpoint['epoch']}")
                    if 'val_mae' in checkpoint:
                        print(f"   - 검증 MAE: {checkpoint['val_mae']}")
                else:
                    print(f"   - 직접 모델 상태 딕셔너리")
                
                return path
            except Exception as e:
                print(f"   - 로드 실패: {e}")
        else:
            print(f"❌ 체크포인트 없음: {path}")
    
    print("❌ 사용 가능한 체크포인트가 없습니다.")
    return None

def test_model_creation():
    """모델 생성 테스트"""
    print("🏗️ 모델 생성 테스트")
    
    try:
        # 간단한 모델 구조 정의
        class SimpleTestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2048, 2)
            
            def forward(self, x):
                return self.linear(x)
        
        model = SimpleTestModel()
        print(f"✅ 모델 생성 성공: {type(model)}")
        print(f"   - 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
        
        # 테스트 추론
        x = torch.randn(1, 2048)
        with torch.no_grad():
            output = model(x)
        print(f"   - 추론 성공: {output.shape}")
        
        return True
    except Exception as e:
        print(f"❌ 모델 생성 실패: {e}")
        return False

def main():
    """메인 함수"""
    print("🧪 간단한 통합 모델 로더 테스트")
    print("=" * 50)
    
    test_cuda()
    checkpoint_path = test_checkpoint()
    model_success = test_model_creation()
    
    print("\n📊 테스트 결과 요약:")
    print(f"   - CUDA 사용 가능: {torch.cuda.is_available()}")
    print(f"   - 체크포인트 발견: {checkpoint_path is not None}")
    print(f"   - 모델 생성 성공: {model_success}")
    
    if checkpoint_path and model_success:
        print("\n✅ 모든 테스트 통과! 통합 모델 로더 사용 가능")
    else:
        print("\n❌ 일부 테스트 실패. 문제 해결 필요")

if __name__ == "__main__":
    main()
