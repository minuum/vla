#!/bin/bash
# Mobile VLA LoRA Fine-tuning 실행 스크립트 (2025-11-14)
# 기존 RoboVLMs 코드 기반으로 실행

set -e

echo "========================================="
echo "Mobile VLA LoRA Fine-tuning"
echo "Date: 2025-11-14"
echo "========================================="

# 작업 디렉토리 확인
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# RoboVLMs 서브모듈 디렉토리로 이동
cd RoboVLMs_upstream

# CUDA 설정
export CUDA_VISIBLE_DEVICES=0
export RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=localhost
export MASTER_PORT=29500

# Config 경로
CONFIG="../Mobile_VLA/configs/mobile_vla_20251114_lora.json"

echo ""
echo "📄 Config: $CONFIG"
echo "🔧 Device: CUDA"
echo "📦 Model: Kosmos-2 with LoRA"
echo ""

# CUDA 확인
echo "🔍 CUDA 확인..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo ""

# 데이터셋 확인
echo "📊 데이터셋 확인..."
EPISODE_COUNT=$(find ../ROS_action/mobile_vla_dataset -name "episode_2025111*.h5" | wc -l)
echo "  - Found $EPISODE_COUNT episodes matching pattern 'episode_2025111*.h5'"
if [ "$EPISODE_COUNT" -eq 0 ]; then
    echo "  ⚠️  Warning: No episodes found! Check episode_pattern in config."
else
    echo "  ✅ Episodes found"
fi
echo ""

# 설정 파일 확인
if [ ! -f "$CONFIG" ]; then
    echo "❌ Error: Config file not found: $CONFIG"
    exit 1
fi

echo "🚀 LoRA Fine-tuning 시작..."
echo "   - Using RoboVLMs main.py"
echo "   - Dataset: MobileVLAH5Dataset"
echo "   - LoRA: r=32, alpha=16, dropout=0.1"
echo "   - Epochs: 20"
echo ""

# 학습 시작
python3 main.py "$CONFIG"

echo ""
echo "✅ LoRA Fine-tuning 완료!"
echo ""
echo "📁 결과 확인:"
echo "   - Checkpoints: runs/mobile_vla_lora_20251114/checkpoints/"
echo "   - Logs: runs/mobile_vla_lora_20251114/logs/"
echo ""

