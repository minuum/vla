#!/bin/bash
# Mobile VLA - RoboVLMs Frozen+LoRA 학습 (Poetry 환경)
# 2025-12-04

echo "========================================="
echo "🚀 RoboVLMs Frozen+LoRA 학습 시작"
echo "GPU 학습: Poetry 환경 사용"
echo "========================================="
echo ""

CONFIG="../Mobile_VLA/configs/mobile_vla_robovlms_frozen_lora_20251204.json"

# CUDA 확인
echo "🔍 CUDA 확인..."
if poetry run python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    GPU_NAME=$(poetry run python -c "import torch; print(torch.cuda.get_device_name() if torch.cuda.is_available() else 'N/A')")
    echo "  ✅ GPU: $GPU_NAME"
else
    echo "  ❌ CUDA를 사용할 수 없습니다."
    exit 1
fi

# Poetry 환경 확인
echo ""
echo "🔍 Poetry 환경 확인..."
poetry env info || {
    echo "  ❌ Poetry 환경 없음"
    exit 1
}

# 타임스탬프
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="../lora_training_robovlms_${TIMESTAMP}.txt"

echo ""
echo "📝 Config: mobile_vla_robovlms_frozen_lora_20251204.json"
echo "📄 Log: $LOG_FILE"
echo ""
echo "🎯 목적: Robot pretrain VLM vs 일반 VLM 비교"
echo ""
echo "========================================="
echo "학습 시작..."
echo "========================================="
echo ""

# Poetry 환경에서 학습 실행
cd /home/soda/25-1kp/vla/RoboVLMs_upstream
poetry run python main.py $CONFIG 2>&1 | tee $LOG_FILE

echo ""
echo "========================================="
echo "✅ 학습 완료"
echo "📄 로그: $LOG_FILE"
echo "========================================="
