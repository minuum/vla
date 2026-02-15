#!/usr/bin/env python3
import torch
import sys

def main():
    print("🔍 PyTorch & CUDA 테스트")
    print(f"PyTorch 버전: {torch.__version__}")
    print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA 버전: {torch.version.cuda}")
        print(f"CUDA 디바이스 수: {torch.cuda.device_count()}")
        print(f"현재 CUDA 디바이스: {torch.cuda.current_device()}")
        print(f"디바이스 이름: {torch.cuda.get_device_name(0)}")
        
        # 간단한 CUDA 연산 테스트
        try:
            x = torch.rand(5, 3).cuda()
            y = torch.rand(5, 3).cuda()
            z = x + y
            print("✅ CUDA 연산 테스트 성공")
        except Exception as e:
            print(f"❌ CUDA 연산 테스트 실패: {e}")
    else:
        print("❌ CUDA를 사용할 수 없습니다")
    
    print("🎉 테스트 완료")

if __name__ == "__main__":
    main()
