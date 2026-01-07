#!/bin/bash
# Batch quantization for all best models
# PTQ (Post-Training Quantization) for comparison

set -e

echo "================================================================"
echo "🚀 Batch PTQ Quantization for Mobile VLA Models"
echo "================================================================"

# Output base directory
OUTPUT_BASE="quantized_models/batch_ptq_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_BASE"

# Best models to quantize
declare -A MODELS=(
    ["left_chunk10"]="runs/mobile_vla_no_chunk_20251209/kosmos/mobile_vla_finetune/2025-12-18/mobile_vla_left_chunk10_20251218/epoch_epoch=09-val_loss=val_loss=0.010.ckpt"
    ["left_chunk5"]="runs/mobile_vla_no_chunk_20251209/kosmos/mobile_vla_finetune/2025-12-18/mobile_vla_left_chunk5_20251218/epoch_epoch=08-val_loss=val_loss=0.016.ckpt"
    ["right_chunk10"]="runs/mobile_vla_no_chunk_20251209/kosmos/mobile_vla_finetune/2025-12-18/mobile_vla_right_chunk10_20251218/epoch_epoch=09-val_loss=val_loss=0.013.ckpt"
    ["chunk5"]="runs/mobile_vla_no_chunk_20251209/kosmos/mobile_vla_finetune/2025-12-17/mobile_vla_chunk5_20251217/epoch_epoch=06-val_loss=val_loss=0.067.ckpt"
    ["chunk10"]="runs/mobile_vla_no_chunk_20251209/kosmos/mobile_vla_finetune/2025-12-17/mobile_vla_chunk10_20251217/epoch_epoch=05-val_loss=val_loss=0.284.ckpt"
)

declare -A CONFIGS=(
    ["left_chunk10"]="Mobile_VLA/configs/mobile_vla_left_chunk10_20251218.json"
    ["left_chunk5"]="Mobile_VLA/configs/mobile_vla_left_chunk5_20251218.json"
    ["right_chunk10"]="Mobile_VLA/configs/mobile_vla_right_chunk10_20251218.json"
    ["chunk5"]="Mobile_VLA/configs/mobile_vla_chunk5_20251217.json"
    ["chunk10"]="Mobile_VLA/configs/mobile_vla_chunk10_20251217.json"
)

# Calibration data directory
DATA_DIR="ROS_action/mobile_vla_dataset"

# Summary file
SUMMARY_FILE="$OUTPUT_BASE/quantization_summary.txt"
echo "Batch Quantization Summary" > "$SUMMARY_FILE"
echo "Date: $(date)" >> "$SUMMARY_FILE"
echo "======================================" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

# Quantize each model
for model_name in "${!MODELS[@]}"; do
    echo ""
    echo "================================================================"
    echo "Processing: $model_name"
    echo "================================================================"
    
    
    checkpoint="${MODELS[$model_name]}"
    config="${CONFIGS[$model_name]}"
    output_dir="$OUTPUT_BASE/$model_name"
    
    # Check if checkpoint exists
    if [ ! -f "$checkpoint" ]; then
        echo "❌ Checkpoint not found: $checkpoint"
        echo "$model_name: SKIPPED (checkpoint not found)" >> "$SUMMARY_FILE"
        continue
    fi
    
    # Create output directory BEFORE tee
    mkdir -p "$output_dir"
    
    echo "📦 Checkpoint: $checkpoint"
    echo "⚙️  Config: $config"
    echo "📁 Output: $output_dir"
    echo ""
    
    # Run quantization
    start_time=$(date +%s)
    
    if python3 scripts/quantize_for_jetson.py \
        --checkpoint "$checkpoint" \
        --config "$config" \
        --data-dir "$DATA_DIR" \
        --vision-int8 \
        --llm-int4 \
        --calib-size 100 \
        --output "$output_dir" 2>&1 | tee "$output_dir/quantization.log"; then
        
        end_time=$(date +%s)
        elapsed=$((end_time - start_time))
        
        echo "✅ $model_name completed in ${elapsed}s"
        echo "$model_name: SUCCESS (${elapsed}s)" >> "$SUMMARY_FILE"
        
        # Extract memory info
        if [ -f "$output_dir/model_info.json" ]; then
            echo "  Memory info:" >> "$SUMMARY_FILE"
            cat "$output_dir/model_info.json" | python3 -m json.tool >> "$SUMMARY_FILE"
            echo "" >> "$SUMMARY_FILE"
        fi
    else
        echo "❌ $model_name failed"
        echo "$model_name: FAILED" >> "$SUMMARY_FILE"
    fi
    
    echo ""
done

echo ""
echo "================================================================"
echo "✅ Batch Quantization Complete!"
echo "================================================================"
echo ""
echo "📁 Output directory: $OUTPUT_BASE"
echo "📊 Summary: $SUMMARY_FILE"
echo ""

# Display summary
cat "$SUMMARY_FILE"

echo ""
echo "💡 Next steps:"
echo "  1. Review quantization logs"
echo "  2. Run validation:"
echo "     bash scripts/validate_all_quantized.sh $OUTPUT_BASE"
echo "  3. Compare results"
echo ""
