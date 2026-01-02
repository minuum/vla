#!/bin/bash
# Case 3: Kosmos-2 Frozen+LoRA, Left+Right 균형 데이터
# 2025-12-04

echo "========================================="
echo "🚀 Case 3 학습 시작"
echo "========================================="
echo ""
echo "실험: Kosmos-2 Frozen+LoRA + Left+Right"
echo "데이터: 500 episodes (250 left + 250 right)"
echo ""

CONFIG="../Mobile_VLA/configs/mobile_vla_kosmos2_frozen_lora_leftright_20251204.json"

# CUDA 확인
echo "🔍 GPU 확인..."
if python3 -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name() if torch.cuda.is_available() else 'N/A')")
    echo "  ✅ GPU: $GPU_NAME"
else
    echo "  ❌ CUDA 사용 불가"
    exit 1
fi

# 타임스탬프
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="../case3_kosmos2_leftright_${TIMESTAMP}.txt"

echo ""
echo "📝 Config: mobile_vla_kosmos2_frozen_lora_leftright_20251204.json"
echo "📄 Log: $LOG_FILE"
echo ""
echo "========================================="
echo "학습 시작..."
echo "========================================="
echo ""

# 학습 실행
cd /home/soda/25-1kp/vla/RoboVLMs_upstream
python3 main.py $CONFIG 2>&1 | tee $LOG_FILE

echo ""
echo "========================================="
echo "✅ Case 3 학습 완료"
echo "📄 로그: $LOG_FILE"
echo "========================================="
