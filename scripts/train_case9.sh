#!/bin/bash
# Updated to RoboVLM-Nav structure

set -e
PROJECT_ROOT="/home/billy/25-1kp/vla"
CONFIG="$PROJECT_ROOT/configs/mobile_vla_no_chunk_aug_abs_20251210.json"

echo "🚀 [RoboVLM-Nav] Running training with config: $CONFIG"

export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/third_party/RoboVLMs"
cd $PROJECT_ROOT

python3 robovlm_nav/train.py $CONFIG
