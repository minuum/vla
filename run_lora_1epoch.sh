#!/bin/bash
# Mobile VLA LoRA Fine-tuning 1 Epoch

set -e

echo "========================================="
echo "Mobile VLA LoRA Fine-tuning (1 Epoch)"
echo "========================================="

cd /home/billy/25-1kp/vla/RoboVLMs_upstream

# CUDA 설정
export CUDA_VISIBLE_DEVICES=0
export RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=localhost
export MASTER_PORT=29500

# Config 경로
CONFIG="../Mobile_VLA/configs/mobile_vla_20251106_lora.json"

echo "📄 Config: $CONFIG"
echo "🔧 Device: CUDA"
echo ""

# 데이터셋 확인
echo "📊 데이터셋 확인..."
ls -1 ../ROS_action/mobile_vla_dataset/episode_20251106_*.h5 | wc -l
echo ""

# LoRA Fine-tuning 시작
echo "🚀 LoRA Fine-tuning 시작..."
python3 main.py $CONFIG

echo ""
echo "✅ LoRA Fine-tuning 완료!"
