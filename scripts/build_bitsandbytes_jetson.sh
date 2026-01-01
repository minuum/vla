#!/bin/bash
#
# BitsAndBytes 소스 빌드 스크립트 (Jetson Orin)
# 참고: https://forums.developer.nvidia.com/t/bitsandbytes-on-nvidia-jetson-agx-orin/338248/2
#

set -e

echo "======================================================================="
echo "  BitsAndBytes 소스 빌드 (Jetson Orin)"
echo "======================================================================="
echo ""

# 1. Prerequisites 확인
echo "1️⃣ Prerequisites 확인..."
echo ""

echo "Python 버전:"
python3 --version

echo ""
echo "CUDA 버전:"
nvcc --version 2>/dev/null || echo "⚠️  nvcc not found (정상 - PyTorch CUDA 사용)"

echo ""
echo "PyTorch CUDA:"
python3 -c "import torch; print(f'  PyTorch: {torch.__version__}'); print(f'  CUDA: {torch.version.cuda}'); print(f'  Available: {torch.cuda.is_available()}')"

echo ""
echo "======================================================================="
echo ""

# 2. 기존 bitsandbytes 제거 (선택)
echo "2️⃣ 기존 bitsandbytes 백업..."
INSTALLED_VERSION=$(pip show bitsandbytes 2>/dev/null | grep Version | awk '{print $2}')

if [ -n "$INSTALLED_VERSION" ]; then
    echo "   현재 설치됨: bitsandbytes $INSTALLED_VERSION"
    echo "   백업 방법: pip freeze > bitsandbytes_backup.txt"
    echo ""
    read -p "   기존 버전을 제거하고 소스 빌드를 진행하시겠습니까? (y/N): " -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "   제거 중..."
        pip uninstall -y bitsandbytes
    else
        echo "   건너뜀. 기존 버전 유지."
        exit 0
    fi
else
    echo "   bitsandbytes 미설치"
fi

echo ""
echo "======================================================================="
echo ""

# 3. 소스 클론
echo "3️⃣ BitsAndBytes 소스 클론..."
REPO_DIR="/tmp/bitsandbytes_build"

if [ -d "$REPO_DIR" ]; then
    echo "   기존 디렉토리 삭제..."
    rm -rf "$REPO_DIR"
fi

git clone https://github.com/TimDettmers/bitsandbytes.git "$REPO_DIR"
cd "$REPO_DIR"

# 특정 버전 체크아웃 (선택)
echo ""
read -p "   특정 버전을 체크아웃하시겠습니까? (0.43.1 권장) [0.43.1/latest]: " VERSION
VERSION=${VERSION:-latest}

if [ "$VERSION" != "latest" ]; then
    echo "   체크아웃: $VERSION"
    git checkout "$VERSION"
fi

echo ""
echo "======================================================================="
echo ""

# 4. 빌드 환경 설정
echo "4️⃣ 빌드 환경 설정..."
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

echo "   CUDA_HOME: $CUDA_HOME"
echo "   PATH: $PATH"
echo ""

# 5. 의존성 설치
echo "5️⃣ 의존성 설치..."
if [ -f "requirements.txt" ]; then
    pip3 install -r requirements.txt
else
    echo "   requirements.txt 없음, 건너뜀"
fi

echo ""
echo "======================================================================="
echo ""

# 6. 빌드 & 설치
echo "6️⃣ 빌드 및 설치 (시간이 걸릴 수 있습니다...)"
echo "   ⏳ 예상 시간: 10-30분"
echo ""

# Jetson용 빌드 플래그
export CUDA_VERSION=122  # CUDA 12.2
export COMPUTE_CAPABILITY=87  # Orin = 8.7

python3 setup.py install --user

echo ""
echo "======================================================================="
echo ""

# 7. 설치 확인
echo "7️⃣ 설치 확인..."
python3 << 'PYTHON_TEST'
try:
    import bitsandbytes as bnb
    print(f"✅ BitsAndBytes 버전: {bnb.__version__}")
    print(f"   설치 경로: {bnb.__file__}")
    
    # CUDA 지원 확인
    import torch
    from transformers import BitsAndBytesConfig
    
    config = BitsAndBytesConfig(load_in_8bit=True)
    print(f"✅ BitsAndBytesConfig 생성 성공")
    
    print("\n🎉 소스 빌드 성공!")
    
except Exception as e:
    print(f"❌ 실패: {e}")
    import traceback
    traceback.print_exc()
PYTHON_TEST

echo ""
echo "======================================================================="
echo "✅ 빌드 완료!"
echo "======================================================================="
