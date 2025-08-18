#!/usr/bin/env bash
set -euo pipefail

# Mobile VLA + Kosmos 학습 스크립트

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)

export MOBILE_VLA_DATA_DIR=${MOBILE_VLA_DATA_DIR:-"$ROOT_DIR/../../ROS_action/mobile_vla_dataset"}

echo "🤖 Mobile VLA + Kosmos 학습 시작"
echo "📁 데이터 디렉토리: $MOBILE_VLA_DATA_DIR"

# Kosmos 학습 실행
python3 "$ROOT_DIR/training/train_kosmos_mobile.py" \
  --data_dir "$MOBILE_VLA_DATA_DIR" \
  --batch_size 1 \
  --sequence_length 18 \
  --hidden_size 768 \
  --max_steps 10 \
  --learning_rate 1e-4 \
  --freeze_kosmos \
  --kosmos_model "microsoft/kosmos-2-patch14-224"

echo "✅ Mobile VLA + Kosmos 학습 완료"
