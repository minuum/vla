#!/bin/bash
# QAT 학습 스크립트 - Unified Model (Left + Right)
# INT8 Vision Encoder + INT4 LLM
# 500개 데이터 (20251203-04)

set -e

echo "=================================================="
echo "🚀 QAT Training - Unified Model (Left + Right)"
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
LOG_FILE="$LOG_DIR/train_qat_unified_chunk10_${TIMESTAMP}.log"

echo "📋 Configuration:"
echo "  - Config: Mobile_VLA/configs/mobile_vla_qat_unified_chunk10_20251223.json"
echo "  - Data: ROS_action/mobile_vla_dataset (500 episodes: 20251203-04)"
echo "  - Left + Right: Unified model"
echo "  - Trainer: MobileVLAQATTrainer"
echo "  - Vision: INT8 (QAT)"
echo "  - LLM: INT4 (BitsAndBytes)"
echo "  - Action Head: FP16 (Trainable)"
echo "  - Chunk Size: 10"
echo "  - Log: $LOG_FILE"
echo ""

# 학습 실행
echo "🔥 Starting QAT training..."
python3 RoboVLMs_upstream/main.py \
    Mobile_VLA/configs/mobile_vla_qat_unified_chunk10_20251223.json \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "=================================================="
echo "✅ QAT Training Complete!"
echo "=================================================="
echo "📁 Log saved to: $LOG_FILE"
echo "📊 Checkpoints: runs/mobile_vla_qat_20251223/"
echo ""
