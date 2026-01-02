#!/bin/bash
# Quick start script for quantization
# Usage: bash scripts/quick_quantize.sh

cd /home/soda/25-1kp/vla

echo "🚀 Starting INT8/INT4 Quantization..."
echo ""

# Best model: Chunk5 Epoch 6
CHECKPOINT="runs/mobile_vla_no_chunk_20251209/kosmos/mobile_vla_finetune/2025-12-17/mobile_vla_chunk5_20251217/epoch_epoch=06-val_loss=val_loss=0.067.ckpt"
CONFIG="Mobile_VLA/configs/mobile_vla_chunk5_20251217.json"
OUTPUT="quantized_models/chunk5_int8_int4_$(date +%Y%m%d)"

python3 scripts/quantize_for_jetson.py \
    --checkpoint "$CHECKPOINT" \
    --config "$CONFIG" \
    --data-dir ROS_action/mobile_vla_dataset \
    --vision-int8 \
    --llm-int4 \
    --calib-size 100 \
    --output "$OUTPUT"

echo ""
echo "✅ Quantization completed!"
echo "📁 Output: $OUTPUT"
