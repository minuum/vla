#!/bin/bash
# Case 4: RIGHT ONLY 학습
# 박스를 오른쪽으로 피해가는 에피소드만 학습
# 2025-12-07

echo "========================================="
echo "🚀 Case 4 학습 시작 (RIGHT ONLY)"
echo "========================================="
echo ""
echo "실험: Kosmos-2 Frozen+LoRA + Right Only"
echo "데이터: 250 episodes (right only)"
echo "태스크: 장애물(박스)을 오른쪽으로 피해서 목표물로 이동"
echo ""

CONFIG="../Mobile_VLA/configs/mobile_vla_kosmos2_right_only_20251207.json"

# CUDA 확인
echo "🔍 GPU 확인..."
if python3 -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name() if torch.cuda.is_available() else 'N/A')")
    echo "  ✅ GPU: $GPU_NAME"
else
    echo "  ❌ CUDA 사용 불가"
    exit 1
fi

# Right 파일 개수 확인
RIGHT_FILES=$(ls /home/soda/25-1kp/vla/ROS_action/mobile_vla_dataset/*right*.h5 2>/dev/null | wc -l)
echo "  📁 Right episodes: $RIGHT_FILES"

# 타임스탬프
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="../case4_kosmos2_right_only_${TIMESTAMP}.txt"

echo ""
echo "📝 Config: mobile_vla_kosmos2_right_only_20251207.json"
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
echo "✅ Case 4 학습 완료"
echo "📄 로그: $LOG_FILE"
echo "========================================="
