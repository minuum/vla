#!/bin/bash

# V3-EXP-05 LoRA Training Script
# 목적: Forward collapse 해결 - Left/FwdLeft 가중치 강화 (역빈도 기반)
# 변경점 vs exp04:
#   - Class 1 (Fwd) weight: 0.2 -> 0.1  (Forward 더 억제)
#   - Class 3 (L)   weight: 5.0 -> 10.0 (Left 2배 강화)
#   - Class 5 (FL)  weight: 5.0 -> 1.5  (FL 충분히 있으므로 낮춤)
#   - Class 6 (FR)  weight: 5.0 -> 1.0  (FR 가장 많으므로 낮춤)
#   - max_epochs: 10 -> 12

export CUDA_VISIBLE_DEVICES=0

mkdir -p logs

cd RoboVLMs_upstream

python3 -u main.py ../Mobile_VLA/configs/mobile_vla_v3_exp05_lora.json 2>&1 | tee ../logs/train_v3_exp05_lora.log

echo "✅ V3-EXP-05 LoRA Training Completed!"
