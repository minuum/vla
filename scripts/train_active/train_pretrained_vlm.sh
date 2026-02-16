#!/bin/bash
# RoboVLMs Pretrained VLM 기반 Mobile VLA 학습
# VLM: Google Robot pretrained (Frozen)
# Action Head: 2DoF Mobile VLA (새로 학습)

set -e

echo "======================================"
echo "RoboVLMs Pretrained VLM Training"
echo "======================================"
echo ""
echo "Config: Mobile_VLA/configs/mobile_vla_pretrained.json"
echo "Pretrained VLM: pretrained_ckpts/checkpoints/kosmos_ph_google-robot-post-train.pt"
echo "VLM Status: FROZEN (no fine-tuning)"
echo "Action Head: 2DoF Mobile VLA (NEW)"
echo ""

cd /home/billy/25-1kp/vla

# GPU 메모리 정리
nvidia-smi

echo ""
echo "Starting training..."
echo ""

python3 RoboVLMs_upstream/main.py \
    Mobile_VLA/configs/mobile_vla_pretrained.json
