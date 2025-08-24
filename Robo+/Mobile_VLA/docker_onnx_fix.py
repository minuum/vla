#!/usr/bin/env python3
"""
Docker 컨테이너 ONNX Runtime 설치 문제 해결
"""

import subprocess
import sys
import os

def run_docker_command(command: str) -> bool:
    """Docker 명령어 실행"""
    try:
        print(f"🔧 Running: {command}")
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Success: {result.stdout}")
            return True
        else:
            print(f"❌ Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

def install_onnx_runtime():
    """ONNX Runtime 설치 시도"""
    print("🚀 Installing ONNX Runtime in Docker container")
    print("=" * 60)
    
    # 방법 1: CPU 버전 설치
    print("\n1️⃣ Trying ONNX Runtime CPU version...")
    success1 = run_docker_command(
        'docker exec -it mobile_vla_robovlms_final bash -c "pip install onnxruntime"'
    )
    
    if not success1:
        # 방법 2: GPU 버전 설치
        print("\n2️⃣ Trying ONNX Runtime GPU version...")
        success2 = run_docker_command(
            'docker exec -it mobile_vla_robovlms_final bash -c "pip install onnxruntime-gpu"'
        )
        
        if not success2:
            # 방법 3: 특정 버전 설치
            print("\n3️⃣ Trying specific ONNX Runtime version...")
            success3 = run_docker_command(
                'docker exec -it mobile_vla_robovlms_final bash -c "pip install onnxruntime==1.17.0"'
            )
            
            if not success3:
                # 방법 4: conda 사용
                print("\n4️⃣ Trying conda installation...")
                success4 = run_docker_command(
                    'docker exec -it mobile_vla_robovlms_final bash -c "conda install -c conda-forge onnxruntime"'
                )
                
                if not success4:
                    print("\n❌ All installation methods failed!")
                    return False
    
    print("\n✅ ONNX Runtime installation completed!")
    return True

def test_onnx_runtime():
    """ONNX Runtime 테스트"""
    print("\n🧪 Testing ONNX Runtime...")
    print("=" * 60)
    
    test_script = '''
import onnxruntime as ort
print("✅ ONNX Runtime imported successfully!")
print(f"Version: {ort.__version__}")
print(f"Available providers: {ort.get_available_providers()}")
'''
    
    success = run_docker_command(
        f'docker exec -it mobile_vla_robovlms_final bash -c "python3 -c \'{test_script}\'"'
    )
    
    return success

def check_docker_status():
    """Docker 컨테이너 상태 확인"""
    print("🔍 Checking Docker container status...")
    print("=" * 60)
    
    # 컨테이너 상태 확인
    run_docker_command("docker ps | grep mobile_vla_robovlms_final")
    
    # 컨테이너 내부 확인
    run_docker_command(
        'docker exec -it mobile_vla_robovlms_final bash -c "ls -la /workspace/vla/"'
    )

def create_alternative_solution():
    """대안 해결책 생성"""
    print("\n💡 Creating alternative solution...")
    print("=" * 60)
    
    # PyTorch만 사용하는 버전 생성
    alternative_code = '''
# ONNX Runtime 없이 PyTorch만 사용하는 버전
import torch
import torch.nn as nn

class SimplePyTorchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(3*224*224, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.action_predictor = nn.Linear(256, 3)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        features = self.feature_extractor(x)
        actions = self.action_predictor(features)
        return actions

# 모델 인스턴스 생성
model = SimplePyTorchModel()
print("✅ Simple PyTorch model created successfully!")
'''
    
    # 파일로 저장
    with open("Robo+/Mobile_VLA/simple_pytorch_model.py", "w") as f:
        f.write(alternative_code)
    
    print("✅ Alternative PyTorch-only solution created!")

def main():
    """메인 함수"""
    print("🚀 Docker ONNX Runtime Fix Script")
    print("🎯 Solving ONNX Runtime installation issues")
    
    # 1. Docker 상태 확인
    check_docker_status()
    
    # 2. ONNX Runtime 설치 시도
    if install_onnx_runtime():
        # 3. 설치 테스트
        if test_onnx_runtime():
            print("\n🎉 ONNX Runtime installation and test successful!")
        else:
            print("\n⚠️ Installation succeeded but test failed!")
            create_alternative_solution()
    else:
        print("\n❌ ONNX Runtime installation failed!")
        create_alternative_solution()
    
    print("\n📋 Summary:")
    print("- Docker container: mobile_vla_robovlms_final")
    print("- Issue: ONNX Runtime not installed")
    print("- Solution: Multiple installation methods attempted")
    print("- Alternative: PyTorch-only model created")

if __name__ == "__main__":
    main()
