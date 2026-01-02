#!/bin/bash
# Mobile VLA 개선된 샘플링으로 재학습 (2025-12-04)

echo "========================================="
echo "📦 Mobile VLA - Improved Sampling 학습"
echo "Date: 2025-12-04"
echo "========================================="
echo ""

CONFIG_FILE="../Mobile_VLA/configs/mobile_vla_20251204_improved_sampling.json"

echo "🔄 샘플링 개선 내용:"
echo "  - Random temporal sampling (에피소드당)"
echo "  - Random start frame (시간적 편향 제거)"
echo "  - 에피소드 간 다양성 증가"
echo ""

echo "📊 데이터셋: 250 episodes (Dec 2025)"
echo "🎯 목표: 일반화 성능 향상"
echo ""

echo "🔍 CUDA 확인..."
if python3 -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name() if torch.cuda.is_available() else 'N/A')")
    echo "  ✅ GPU: $GPU_NAME"
else
    echo "  ❌ CUDA를 사용할 수 없습니다."
    exit 1
fi
echo ""

# 타임스탬프 생성
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="lora_training_improved_sampling_${TIMESTAMP}.txt"

echo "🚀 개선된 샘플링으로 학습 시작..."
echo "📄 로그: $LOG_FILE"
echo ""

# 학습 시작
cd /home/soda/25-1kp/vla/RoboVLMs_upstream
python3 main.py $CONFIG_FILE 2>&1 | tee ../$LOG_FILE

echo ""
echo "========================================="
echo "✅ 학습 완료"
echo "📄 로그: $LOG_FILE"
echo "========================================="
