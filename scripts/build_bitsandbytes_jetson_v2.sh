#!/bin/bash
#
# BitsAndBytes 소스 빌드 (Jetson Orin)
# nvcc 경로: /usr/local/cuda-12.2/bin/nvcc
#

set -e

echo "======================================================================="
echo "  BitsAndBytes 소스 빌드 (Jetson Orin)"
echo "  참고: NVIDIA Forum + jtop 정보"
echo "======================================================================="
echo ""

# CUDA 환경 변수 설정
export CUDA_HOME=/usr/local/cuda-12.2
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

echo "📋 환경 확인:"
echo "   CUDA_HOME: $CUDA_HOME"
echo "   nvcc: $(which nvcc)"
nvcc --version | grep "release"
echo ""

# Python 및 PyTorch 확인
echo "   Python: $(python3 --version)"
python3 -c "import torch; print(f'   PyTorch: {torch.__version__}'); print(f'   CUDA: {torch.version.cuda}')"
echo ""

# 기존 bitsandbytes 버전 확인
CURRENT_VERSION=$(pip show bitsandbytes 2>/dev/null | grep Version | awk '{print $2}')
if [ -n "$CURRENT_VERSION" ]; then
    echo "⚠️  현재 설치된 버전: $CURRENT_VERSION"
    echo "   이 버전을 제거하고 소스 빌드로 대체합니다."
    echo ""
    pip uninstall -y bitsandbytes
else
    echo "✅ bitsandbytes 미설치 (새로 빌드)"
    echo ""
fi

# 소스 클론
REPO_DIR="/tmp/bitsandbytes_jetson"
echo "📦 소스 클론: $REPO_DIR"

if [ -d "$REPO_DIR" ]; then
    rm -rf "$REPO_DIR"
fi

git clone https://github.com/TimDettmers/bitsandbytes.git "$REPO_DIR"
cd "$REPO_DIR"

# 특정 버전 선택
echo ""
echo "🔖 버전 선택:"
echo "   1) latest (main)"
echo "   2) 0.43.1 (안정)"
echo "   3) 0.44.0"
echo ""
read -p "선택 [1-3, 기본 2]: " VERSION_CHOICE
VERSION_CHOICE=${VERSION_CHOICE:-2}

case $VERSION_CHOICE in
    1)
        echo "   → latest (main)"
        ;;
    2)
        echo "   → 0.43.1"
        git checkout 0.43.1 2>/dev/null || git checkout tags/0.43.1 2>/dev/null || echo "   ⚠️ 0.43.1 태그 없음, main 사용"
        ;;
    3)
        echo "   → 0.44.0"
        git checkout 0.44.0 2>/dev/null || git checkout tags/0.44.0 2>/dev/null || echo "   ⚠️ 0.44.0 태그 없음, main 사용"
        ;;
esac

echo ""
echo "======================================================================="
echo ""

# 빌드 환경 변수
export CUDA_VERSION=122  # CUDA 12.2
export COMPUTE_CAPABILITY=87  # Orin Arch 8.7

echo "🔧 빌드 설정:"
echo "   CUDA_VERSION: $CUDA_VERSION"
echo "   COMPUTE_CAPABILITY: $COMPUTE_CAPABILITY"
echo ""

# 의존성 설치
if [ -f "requirements.txt" ]; then
    echo "📦 의존성 설치..."
    pip3 install -r requirements.txt
else
    echo "   requirements.txt 없음, 건너뜀"
fi

echo ""
echo "======================================================================="
echo "🔨 빌드 시작 (10-30분 소요)"
echo "======================================================================="
echo ""

# 빌드 & 설치
python3 setup.py install --user

echo ""
echo "======================================================================="
echo "✅ 빌드 완료!"
echo "======================================================================="
echo ""

# 설치 확인
echo "🧪 설치 확인..."
python3 << 'PYTHON_TEST'
try:
    import bitsandbytes as bnb
    print(f"✅ BitsAndBytes 버전: {bnb.__version__}")
    print(f"   경로: {bnb.__file__}")
    
    # CUDA 바이너리 확인
    import os
    bnb_path = os.path.dirname(bnb.__file__)
    cuda_files = [f for f in os.listdir(bnb_path) if f.endswith('.so')]
    print(f"   CUDA 라이브러리: {len(cuda_files)}개")
    for f in cuda_files[:3]:
        print(f"      - {f}")
    
    # BitsAndBytesConfig 테스트
    from transformers import BitsAndBytesConfig
    import torch
    
    config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=False
    )
    print(f"\n✅ BitsAndBytesConfig 생성 성공!")
    print(f"   load_in_8bit: {config.load_in_8bit}")
    
    print("\n🎉 소스 빌드 완전 성공!")
    
except Exception as e:
    print(f"❌ 설치 확인 실패: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
PYTHON_TEST

echo ""
echo "======================================================================="
echo "🎊 완료! 이제 INT8 모델 로딩을 시도할 수 있습니다."
echo "======================================================================="
