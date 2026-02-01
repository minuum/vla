#!/bin/bash
# Quantize Mobile VLA model for Jetson deployment
# Usage: bash scripts/run_quantization.sh

set -e  # Exit on error

echo "=================================="
echo "🚀 Mobile VLA Quantization for Jetson"
echo "=================================="

# 설정
CHECKPOINT_PATH="runs/mobile_vla_no_chunk_20251209/kosmos/mobile_vla_finetune/2025-12-17/mobile_vla_chunk5_20251217/epoch_epoch=05-val_loss=val_loss=0.067.ckpt"
CONFIG_PATH="Mobile_VLA/configs/mobile_vla_chunk5_20251217.json"
DATA_DIR="/home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset"
OUTPUT_DIR="quantized_models/chunk5_int8_int4"

# Checkpoint 존재 확인
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "❌ Error: Checkpoint not found at $CHECKPOINT_PATH"
    echo "   Available checkpoints:"
    find runs/mobile_vla_no_chunk_20251209 -name "*.ckpt" | head -5
    exit 1
fi

echo ""
echo "📋 Configuration:"
echo "  Checkpoint: $CHECKPOINT_PATH"
echo "  Config: $CONFIG_PATH"
echo "  Calibration Data: $DATA_DIR"
echo "  Output: $OUTPUT_DIR"
echo ""

# Step 1: Vision Encoder INT8 양자화
echo "=================================="
echo "Step 1/3: Quantizing Vision Encoder to INT8"
echo "=================================="
echo ""

python scripts/quantize_for_jetson.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --config "$CONFIG_PATH" \
    --data-dir "$DATA_DIR" \
    --vision-int8 \
    --calib-size 100 \
    --output "$OUTPUT_DIR"

echo ""
echo "✅ Vision Encoder INT8 quantization completed"
echo ""

# Step 2: Full quantization (Vision INT8 + LLM INT4)
echo "=================================="
echo "Step 2/3: Quantizing LLM to INT4"
echo "=================================="
echo ""

# Note: LLM INT4는 BitsAndBytes가 필요하므로 먼저 설치 확인
if ! python -c "import bitsandbytes" 2>/dev/null; then
    echo "⚠️  Warning: bitsandbytes not installed"
    echo "   Installing bitsandbytes..."
    pip install bitsandbytes transformers
fi

python scripts/quantize_for_jetson.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --config "$CONFIG_PATH" \
    --data-dir "$DATA_DIR" \
    --vision-int8 \
    --llm-int4 \
    --calib-size 100 \
    --output "$OUTPUT_DIR"

echo ""
echo "✅ Full quantization completed"
echo ""

# Step 3: Validation
echo "=================================="
echo "Step 3/3: Validating quantized model"
echo "=================================="
echo ""

python scripts/validate_quantized_model.py \
    --original "$CHECKPOINT_PATH" \
    --quantized "$OUTPUT_DIR/model_quantized.ckpt" \
    --config "$CONFIG_PATH" \
    --val-data "$DATA_DIR" \
    --num-samples 50 \
    --output "$OUTPUT_DIR/validation_results.json"

echo ""
echo "✅ Validation completed"
echo ""

# Summary
echo "=================================="
echo "📊 Quantization Summary"
echo "=================================="
echo ""

# Model info 출력
if [ -f "$OUTPUT_DIR/model_info.json" ]; then
    echo "Model Information:"
    cat "$OUTPUT_DIR/model_info.json" | python -m json.tool
    echo ""
fi

# Validation results 출력
if [ -f "$OUTPUT_DIR/validation_results.json" ]; then
    echo "Validation Results:"
    cat "$OUTPUT_DIR/validation_results.json" | python -m json.tool
    echo ""
fi

echo "=================================="
echo "✅ All steps completed successfully!"
echo "=================================="
echo ""
echo "📁 Output directory: $OUTPUT_DIR"
echo ""
echo "💡 Next steps:"
echo "  1. Check validation results above"
echo "  2. Deploy to Jetson with:"
echo "     export VLA_USE_QUANTIZATION=true"
echo "     export VLA_QUANTIZED_CHECKPOINT=\"$OUTPUT_DIR/model_quantized.ckpt\""
echo "     export VLA_QUANTIZED_CONFIG=\"$OUTPUT_DIR/config.json\""
echo "     python Mobile_VLA/inference_server.py"
echo ""
