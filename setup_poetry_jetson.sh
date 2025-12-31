#!/bin/bash
# Poetry 환경 설정 스크립트 (Jetson용)

set -e

echo "="*70
echo "  Poetry 환경 설정 (Jetson JetPack 6.0)"
echo "="*70
echo

# 1. Poetry 설치 확인
echo "📦 Poetry 확인..."
if ! command -v poetry &> /dev/null; then
    echo "   Poetry가 설치되지 않았습니다. 설치 중..."
    curl -sSL https://install.python-poetry.org | python3 -
    
    # PATH 추가
    export PATH="$HOME/.local/bin:$PATH"
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
    
    echo "   ✅ Poetry 설치 완료"
else
    echo "   ✅ Poetry 이미 설치됨: $(poetry --version)"
fi
echo

# 2. Poetry 설정
echo "🔧 Poetry 설정..."
poetry config virtualenvs.in-project true  # .venv를 프로젝트 내에 생성
poetry config virtualenvs.prefer-active-python true
echo "   ✅ 설정 완료 (.venv를 프로젝트 내에 생성)"
echo

# 3. 환경 변수 준비
echo "🌍 환경 변수 설정..."

# CUDA 경로 추가
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.2
export PATH=$CUDA_HOME/bin:$PATH

echo "   ✅ CUDA 경로 설정 완료"
echo "      LD_LIBRARY_PATH: /usr/local/cuda-12.2/lib64"
echo "      CUDA_HOME: /usr/local/cuda-12.2"
echo

# 환경 변수를 Poetry 스크립트에 저장
cat > .envrc << 'EOF'
# Jetson CUDA 환경 변수
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.2
export PATH=$CUDA_HOME/bin:$PATH

# VLA 프로젝트 관련
export PYTHONPATH=$PWD:$PWD/src:$PWD/RoboVLMs:$PYTHONPATH
EOF

echo "   💾 .envrc 파일 생성 (환경 변수 저장)"
echo

# 4. PyTorch wheel 다운로드
echo "📥 PyTorch Wheel 다운로드..."

TORCH_URL="https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.3.0-cp310-cp310-linux_aarch64.whl"
TORCHVISION_URL="https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torchvision-0.18.0-cp310-cp310-linux_aarch64.whl"

mkdir -p wheels

if [ ! -f "wheels/torch-2.3.0-cp310-cp310-linux_aarch64.whl" ]; then
    echo "   Downloading torch..."
    wget -q --show-progress -O wheels/torch-2.3.0-cp310-cp310-linux_aarch64.whl "$TORCH_URL" || {
        echo "   ⚠️  다운로드 실패. 수동 설치 필요:"
        echo "      wget $TORCH_URL"
        echo "      mv torch-*.whl wheels/"
    }
fi

if [ ! -f "wheels/torchvision-0.18.0-cp310-cp310-linux_aarch64.whl" ]; then
    echo "   Downloading torchvision..."
    wget -q --show-progress -O wheels/torchvision-0.18.0-cp310-cp310-linux_aarch64.whl "$TORCHVISION_URL" || {
        echo "   ⚠️  다운로드 실패. 수동 설치 필요:"
        echo "      wget $TORCHVISION_URL"
        echo "      mv torchvision-*.whl wheels/"
    }
fi

echo "   ✅ 다운로드 확인 완료"
echo

# 5. Poetry 환경 생성 및 패키지 설치
echo "📦 Poetry 환경 생성 및 패키지 설치..."
echo "   (이 과정은 5-10분 소요될 수 있습니다)"
echo

# pyproject.toml 업데이트 (로컬 wheel 사용)
cat > pyproject_local.toml << 'EOF'
[tool.poetry]
name = "mobile-vla"
version = "1.0.0"
description = "Mobile VLA - Vision-Language-Action Model for Jetson"
authors = ["VLA Team"]
readme = "README.md"

[tool.poetry.dependencies]
python = "^3.10"

# PyTorch - 로컬 wheel 사용
torch = {path = "wheels/torch-2.3.0-cp310-cp310-linux_aarch64.whl"}
torchvision = {path = "wheels/torchvision-0.18.0-cp310-cp310-linux_aarch64.whl"}

# Transformers & BitsAndBytes
transformers = "4.41.2"
accelerate = "^0.27.0"
bitsandbytes = "0.42.0"

# Vision & Processing
pillow = "^10.0.0"
opencv-python = "4.5.4"  # Jetson 호환 버전

# Utils
numpy = "^1.24,<2.0"
psutil = "^5.9.0"
tqdm = "^4.65.0"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
EOF

# 원본 백업
if [ -f "pyproject.toml" ]; then
    cp pyproject.toml pyproject.toml.bak
fi

cp pyproject_local.toml pyproject.toml

# 설치 시도
poetry install || {
    echo "   ⚠️  자동 설치 실패"
    echo ""
    echo "   수동 설치 방법:"
    echo "   1. poetry shell"
    echo "   2. pip install wheels/torch-*.whl"
    echo "   3. pip install wheels/torchvision-*.whl"
    echo "   4. pip install transformers==4.41.2 bitsandbytes==0.42.0"
    exit 1
}

echo "   ✅ 패키지 설치 완료"
echo

# 6. 환경 활성화 안내
echo "="*70
echo "✅ Poetry 환경 설정 완료!"
echo "="*70
echo
echo "다음 명령어로 환경을 활성화하세요:"
echo
echo "  source .envrc          # 환경 변수 로드"
echo "  poetry shell           # Poetry 가상환경 활성화"
echo
echo "환경 활성화 후 확인:"
echo
echo "  python -c 'import torch; print(f\"PyTorch: {torch.__version__}\"); print(f\"CUDA: {torch.cuda.is_available()}\")'"
echo
echo "측정 스크립트 실행:"
echo
echo "  python scripts/diagnose_jetson_environment.py"
echo "  python scripts/measure_baseline_memory.py"
echo
