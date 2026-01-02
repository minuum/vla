#!/bin/bash
# Mobile VLA LoRA 학습 시작 스크립트 (2025-12-03)
# 최신 데이터셋 (Nov + Dec 2025)을 사용하여 학습

echo "========================================="
echo "📦 Mobile VLA LoRA Fine-tuning"
echo "Date: 2025-12-03"
echo "========================================="
echo ""

CONFIG_FILE="../Mobile_VLA/configs/mobile_vla_20251203_lora.json"

echo "📄 Config: $CONFIG_FILE"
echo "🔧 Device: CUDA"
echo "📦 Model: Kosmos-2 with LoRA"
echo ""

echo "🔍 CUDA 확인..."
if python3 -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name() if torch.cuda.is_available() else 'N/A')")
    GPU_MEMORY=$(python3 -c "import torch; print(f'{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB' if torch.cuda.is_available() else 'N/A')")
    echo "  ✅ GPU: $GPU_NAME ($GPU_MEMORY)"
else
    echo "  ❌ CUDA를 사용할 수 없습니다."
    exit 1
fi
echo ""

echo "📊 데이터셋 확인..."
EPISODE_COUNT=$(ls -1 /home/soda/25-1kp/vla/ROS_action/mobile_vla_dataset/episode_202511*.h5 2>/dev/null | wc -l)
if [ $EPISODE_COUNT -gt 0 ]; then
    echo "  ✅ $EPISODE_COUNT episodes found (Nov 2025)"
else
    echo "  ⚠️  No Nov 2025 episodes found"
fi

DEC_EPISODE_COUNT=$(ls -1 /home/soda/25-1kp/vla/ROS_action/mobile_vla_dataset/episode_202512*.h5 2>/dev/null | wc -l)
if [ $DEC_EPISODE_COUNT -gt 0 ]; then
    echo "  ✅ $DEC_EPISODE_COUNT episodes found (Dec 2025)"
fi
TOTAL_EPISODES=$((EPISODE_COUNT + DEC_EPISODE_COUNT))
echo "  📈 Total: $TOTAL_EPISODES episodes"
echo ""

echo "🚀 LoRA Fine-tuning 시작..."
echo "   - Using RoboVLMs main.py"
echo "   - Dataset: MobileVLAH5Dataset"
echo "   - LoRA: r=32, alpha=16, dropout=0.1"
echo "   - Epochs: 10"
echo ""

# 타임스탬프 생성
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="lora_training_log_${TIMESTAMP}.txt"

# 학습 시작
cd /home/soda/25-1kp/vla/RoboVLMs_upstream
python3 main.py $CONFIG_FILE 2>&1 | tee ../$LOG_FILE

echo ""
echo "========================================="
echo "✅ 학습 완료"
echo "📄 로그: $LOG_FILE"
echo "========================================="
