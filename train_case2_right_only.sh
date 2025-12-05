#!/bin/bash
# Case 2: Kosmos-2 Frozen+LoRA, Right only
# 2025-12-04

echo "========================================="
echo "🚀 Case 2 학습 시작"
echo "========================================="
echo ""
echo "실험: Kosmos-2 Frozen+LoRA + Right only"
echo "데이터: 250 episodes (right only)"
echo "비교: Case 1 (left) vs Case 2 (right)"
echo ""

CONFIG="../Mobile_VLA/configs/mobile_vla_kosmos2_right_only_20251204.json"

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
LOG_FILE="../case2_kosmos2_right_${TIMESTAMP}.txt"

echo ""
echo "📝 Config: mobile_vla_kosmos2_right_only_20251204.json"
echo "📄 Log: $LOG_FILE"
echo ""
echo "📊 예상 결과:"
echo "  Case 1 (left): Loss ~0.013"
echo "  Case 2 (right): Loss ???"
echo ""
echo "========================================="
echo "학습 시작..."
echo "========================================="
echo ""

# 학습 실행
cd /home/billy/25-1kp/vla/RoboVLMs_upstream
python3 main.py $CONFIG 2>&1 | tee $LOG_FILE

echo ""
echo "========================================="
echo "✅ Case 2 학습 완료"
echo "📄 로그: $LOG_FILE"
echo "========================================="
