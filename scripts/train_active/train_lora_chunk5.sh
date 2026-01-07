#!/bin/bash
# LoRA Fine-tuning with English Instructions
set -e
cd /home/billy/25-1kp/vla

echo "=========================================="
echo "Mobile VLA LoRA Fine-tuning"  
echo "English Instruction + VLM LoRA"
echo "=========================================="

export CUDA_VISIBLE_DEVICES=0

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/train_lora_chunk5_${TIMESTAMP}.log"
mkdir -p logs

CONFIG_FILE="Mobile_VLA/configs/mobile_vla_lora_chunk5.json"

echo ""
echo "Config: $CONFIG_FILE"
echo "Changes from Frozen VLM:"
echo "  - freeze_backbone: false (VLM fine-tuning)"
echo "  - lora_enable: true (LoRA 활성화)"
echo "  - lora_r: 32, lora_alpha: 16"
echo ""
echo "Expected benefits:"
echo "  ✅ VLM learns instruction-specific embeddings"
echo "  ✅ Better instruction grounding"
echo "  ✅ LEFT vs RIGHT distinction"
echo ""
echo "Log: $LOG_FILE"
echo ""

read -p "Start LoRA training? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 1
fi

echo "Starting LoRA fine-tuning..."
python3 RoboVLMs_upstream/main.py "$CONFIG_FILE" > "$LOG_FILE" 2>&1 &

PID=$!
echo "✅ Started (PID: $PID)"
echo ""
echo "Monitor: tail -f $LOG_FILE"
echo "Stop: kill $PID"
