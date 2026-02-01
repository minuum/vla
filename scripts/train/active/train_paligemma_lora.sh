#!/bin/bash
# PaliGemma-3B LoRA Fine-tuning for Mobile VLA
set -e
cd /home/billy/25-1kp/vla

echo "=========================================="
echo "Mobile VLA PaliGemma-3B LoRA Fine-tuning"  
echo "English Instruction + VLM LoRA"
echo "=========================================="

export CUDA_VISIBLE_DEVICES=0

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/train_paligemma_lora_${TIMESTAMP}.log"
mkdir -p logs

CONFIG_FILE="Mobile_VLA/configs/mobile_vla_paligemma_lora.json"

echo ""
echo "Model: PaliGemma-3B"
echo "Config: $CONFIG_FILE"
echo ""
echo "Key Features:"
echo "  ✅ VLM: PaliGemma-3B (2.4B params)"
echo "  ✅ Vision: SigLIP-So400m (efficient)"
echo "  ✅ Language: Gemma-2B (optimized)"
echo "  ✅ LoRA: enabled (rank=16, alpha=32)"
echo "  ✅ Gradient Checkpointing: enabled"
echo "  ✅ Memory: ~12-15 GB (A5000 24GB 가능)"
echo ""
echo "Expected Benefits:"
echo "  - OpenVLA family (검증된 구조)"
echo "  - Better instruction grounding"
echo "  - Memory efficient (Kosmos-2보다 적음)"
echo "  - LEFT vs RIGHT distinction"
echo ""
echo "Training Plan:"
echo "  - Epochs: 10"
echo "  - Batch size: 1 (accumulate 8)"
echo "  - Window size: 8, Chunk size: 5"
echo "  - Learning rate: 0.0001"
echo ""
echo "Log: $LOG_FILE"
echo ""

read -p "Start PaliGemma-3B LoRA training? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 1
fi

echo "Starting PaliGemma-3B LoRA fine-tuning..."
echo "This will take approximately 12-15 hours for 10 epochs."
echo ""

nohup python3 RoboVLMs_upstream/main.py "$CONFIG_FILE" > "$LOG_FILE" 2>&1 &

PID=$!
echo "✅ Started (PID: $PID)"
echo ""
echo "Monitor:"
echo "  - Real-time log: tail -f $LOG_FILE"
echo "  - GPU usage: watch -n 5 nvidia-smi"
echo "  - Training progress: bash scripts/monitor_training.sh"
echo ""
echo "Stop training: kill $PID"
echo ""
echo "Expected checkpoints:"
echo "  runs/mobile_vla_paligemma/paligemma/mobile_vla_paligemma_finetune/..."
echo ""
