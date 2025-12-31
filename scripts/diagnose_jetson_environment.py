#!/usr/bin/env python3
"""
Jetson 환경 진단
라이브러리 버전 및 호환성 체크
"""

import sys
import importlib
import subprocess
from pathlib import Path


def check_python_version():
    """Python 버전 확인"""
    version = sys.version_info
    print(f"🐍 Python: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 8:
        print(f"   ✅ Python 3.8+ (호환)")
    else:
        print(f"   ⚠️  Python 3.8+ 권장")
    print()


def check_library(name, required_version=None, import_name=None):
    """라이브러리 체크"""
    
    if import_name is None:
        import_name = name
    
    try:
        lib = importlib.import_module(import_name)
        version = getattr(lib, '__version__', 'unknown')
        
        status = "✅"
        note = ""
        
        if required_version:
            if version != required_version:
                status = "⚠️"
                note = f" (권장: {required_version})"
        
        print(f"{status} {name:<20} {version:<15} {note}")
        return True, version
        
    except ImportError as e:
        print(f"❌ {name:<20} Not installed")
        return False, None


def check_cuda():
    """CUDA 버전 확인"""
    try:
        result = subprocess.run(
            ['nvcc', '--version'],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            # CUDA 버전 파싱
            output = result.stdout
            for line in output.split('\n'):
                if 'release' in line:
                    print(f"🎮 CUDA: {line.strip()}")
                    break
        else:
            print("❌ CUDA: nvcc not found")
            
    except FileNotFoundError:
        print("❌ CUDA: nvcc not found")
    
    print()


def check_gpu():
    """GPU 정보 확인"""
    try:
        import torch
        
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            
            print(f"🎮 GPU Device: {device_name}")
            print(f"   CUDA Version (PyTorch): {cuda_version}")
            print(f"   PyTorch CUDA Available: ✅")
        else:
            print("⚠️  GPU: CUDA not available in PyTorch")
    except:
        print("❌ GPU: Could not check")
    
    print()


def main():
    """메인 함수"""
    
    print("="*70)
    print("  Jetson 환경 진단")
    print("="*70)
    print()
    
    # Python 버전
    check_python_version()
    
    # CUDA
    check_cuda()
    
    # GPU
    check_gpu()
    
    # 주요 라이브러리
    print("📦 주요 라이브러리:")
    print()
    
    libraries = [
        ("torch", "2.2.2", "torch"),
        ("torchvision", "0.17.2", "torchvision"),
        ("transformers", "4.41.2", "transformers"),
        ("bitsandbytes", "0.43.1", "bitsandbytes"),
        ("accelerate", None, "accelerate"),
        ("numpy", None, "numpy"),
        ("pillow", None, "PIL"),
        ("opencv-python", None, "cv2"),
        ("psutil", None, "psutil"),
    ]
    
    results = {}
    for lib_info in libraries:
        if len(lib_info) == 3:
            name, required, import_name = lib_info
        else:
            name, required = lib_info
            import_name = name
        
        installed, version = check_library(name, required, import_name)
        results[name] = {"installed": installed, "version": version}
    
    print()
    print("="*70)
    
    # 요약
    total = len(libraries)
    installed = sum(1 for r in results.values() if r['installed'])
    
    print(f"✅ {installed}/{total} 라이브러리 설치됨")
    
    not_installed = [name for name, r in results.items() if not r['installed']]
    if not_installed:
        print()
        print("❌ 미설치 라이브러리:")
        for name in not_installed:
            print(f"   - {name}")
        print()
        print("설치 명령어:")
        print(f"   pip install {' '.join(not_installed)}")
    
    print()
    print("="*70)
    print("✅ 진단 완료!")
    print("="*70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
