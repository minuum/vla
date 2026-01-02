#!/bin/bash
# LoRA 재학습 (균형 데이터)
# 목적: 500개 균형 데이터(L/R 50:50)로 LoRA를 공정하게 학습

cd /home/soda/25-1kp/vla/RoboVLMs_upstream

# Python path 설정
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 학습 시작
echo "=== LoRA Balanced Training (500 episodes, L/R 50:50) ==="
echo "Config: mobile_vla_lora_balanced_20251207.json"
echo "Start: $(date)"

python3 main.py "../Mobile_VLA/configs/mobile_vla_lora_balanced_20251207.json" \
    --accelerator gpu \
    --gpus 1 \
    --strategy auto \
    --max_epochs 10 \
    --precision 16-mixed \
    --gradient_clip_val 1.0

echo "End: $(date)"
