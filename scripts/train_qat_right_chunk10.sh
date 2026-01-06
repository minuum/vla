#!/bin/bash
# QAT 학습 스크립트 - Right Turn Model
# INT8 Vision Encoder + INT4 LLM

set -e

echo "=================================================="
echo "🚀 QAT Training - Right Turn Model"
echo "=================================================="
echo ""

# GPU 설정
export CUDA_VISIBLE_DEVICES=0

# 타임스탬프
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Log 디렉토리
LOG_DIR="logs/qat_training"
mkdir -p "$LOG_DIR"

# Log 파일
LOG_FILE="$LOG_DIR/train_qat_right_chunk10_${TIMESTAMP}.log"

echo "📋 Configuration:"
echo "  - Config: Mobile_VLA/configs/mobile_vla_qat_right_chunk10_20251223.json"
echo "  - Data: ROS_action/mobile_vla_dataset (Right turn episodes)"
echo "  - Trainer: MobileVLAQATTrainer"
echo "  - Vision: INT8 (QAT)"
echo "  - LLM: INT4 (BitsAndBytes)"
echo "  - Action Head: FP16 (Trainable)"
echo "  - Log: $LOG_FILE"
echo ""

# 학습 실행
echo "🔥 Starting QAT training..."
python3 RoboVLMs_upstream/robovlms/train/main.py \
    --config Mobile_VLA/configs/mobile_vla_qat_right_chunk10_20251223.json \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "=================================================="
echo "✅ QAT Training Complete!"
echo "=================================================="
echo "📁 Log saved to: $LOG_FILE"
echo "📊 Checkpoints: runs/mobile_vla_qat_20251223/"
echo ""
