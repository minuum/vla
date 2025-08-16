#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)

export MOBILE_VLA_DATA_DIR=${MOBILE_VLA_DATA_DIR:-"$ROOT_DIR/../../ROS_action/mobile_vla_dataset"}

python3 "$ROOT_DIR/training/train_mobile_vla.py" \
  --data_dir "$MOBILE_VLA_DATA_DIR" \
  --batch_size 2 \
  --sequence_length 18 \
  --hidden_size 512 \
  --use_lite_mode \
  --max_steps 5 \
  --num_workers 2 \
  --learning_rate 1e-4


