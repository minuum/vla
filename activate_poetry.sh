#!/bin/bash
# Poetry 환경 활성화 및 PyTorch 설정 스크립트

set -e

echo "="*70
echo "  Poetry 환경 활성화 및 설정"
echo "="*70
echo

# 1. PATH 설정
export PATH="$HOME/.local/bin:$PATH"

# 2. Poetry 확인
if ! command -v poetry &> /dev/null; then
    echo "❌ Poetry가 설치되지 않았습니다."
    exit 1
fi

echo "✅ Poetry 버전: $(poetry --version)"
echo

# 3. 가상환경 확인
if [ ! -d ".venv" ]; then
    echo "⚠️  .venv가 없습니다. poetry install을 먼저 실행하세요."
    exit 1
fi

# 4. 환경 변수 설정 파일 생성
cat > .envrc << 'EOF'
# Jetson CUDA 환경 변수
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH  
export CUDA_HOME=/usr/local/cuda-12.2
export PATH=$CUDA_HOME/bin:$PATH

# VLA 프로젝트
export PYTHONPATH=$PWD:$PWD/src:$PWD/RoboVLMs:$PYTHONPATH

# Poetry PATH
export PATH="$HOME/.local/bin:$PATH"
EOF

echo "✅ .envrc 생성 완료"
echo

# 5. 환경 변수 로드
source .envrc

echo "✅ 환경 변수 로드 완료"
echo

# 6. 사용 방법 안내
echo "="*70
echo "  다음 명령어로 Poetry 환경을 활성화하세요:"
echo "="*70
echo
echo "  source .envrc   # 환경 변수 로드"
echo "  poetry shell    # Poetry 가상환경 활성화"
echo
echo "활성화 후:"
echo
echo "  python --version                              # Python 확인"
echo "  python scripts/diagnose_jetson_environment.py # 환경 진단"
echo
