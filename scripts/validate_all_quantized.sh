#!/bin/bash
# Validate all quantized models and compare with originals
# Usage: bash scripts/validate_all_quantized.sh <quantized_models_dir>

set -e

if [ -z "$1" ]; then
    echo "Usage: bash scripts/validate_all_quantized.sh <quantized_models_dir>"
    exit 1
fi

QUANT_DIR="$1"
RESULTS_DIR="$QUANT_DIR/validation_results"
mkdir -p "$RESULTS_DIR"

echo "================================================================"
echo "🔍 Validating All Quantized Models"
echo "================================================================"
echo ""
echo "Quantized models directory: $QUANT_DIR"
echo "Results directory: $RESULTS_DIR"
echo ""

# Original models and configs
declare -A ORIGINALS=(
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

DATA_DIR="ROS_action/mobile_vla_dataset"
COMPARISON_FILE="$RESULTS_DIR/comparison_summary.md"

# Create comparison summary header
cat > "$COMPARISON_FILE" << 'EOF'
# PTQ Quantization Comparison Summary

**Date**: $(date)  
**Method**: PTQ (Vision INT8 + LLM INT4)

---

## Results

| Model | Original Acc | Quantized Acc | Acc Drop | Latency Original | Latency Quantized | Speedup | Memory (GB) |
|-------|--------------|---------------|----------|------------------|-------------------|---------|-------------|
EOF

# Validate each model
for model_name in "${!ORIGINALS[@]}"; do
    echo ""
    echo "================================================================"
    echo "Validating: $model_name"
    echo "================================================================"
    
    original="${ORIGINALS[$model_name]}"
    config="${CONFIGS[$model_name]}"
    quantized="$QUANT_DIR/$model_name/model_quantized.pt"
    output="$RESULTS_DIR/${model_name}_results.json"
    
    # Check if quantized model exists
    if [ ! -f "$quantized" ]; then
        echo "⚠️  Quantized model not found: $quantized"
        echo "| $model_name | N/A | N/A | N/A | N/A | N/A | N/A | N/A |" >> "$COMPARISON_FILE"
        continue
    fi
    
    echo "Original: $original"
    echo "Quantized: $quantized"
    echo "Config: $config"
    echo ""
    
    # Run validation
    if python3 scripts/validate_quantized_model.py \
        --original "$original" \
        --quantized "$quantized" \
        --config "$config" \
        --val-data "$DATA_DIR" \
        --num-samples 100 \
        --output "$output" 2>&1 | tee "$RESULTS_DIR/${model_name}_validation.log"; then
        
        echo "✅ Validation completed for $model_name"
        
        # Extract results and add to comparison
        if [ -f "$output" ]; then
            python3 << EOF
import json
with open('$output', 'r') as f:
    data = json.load(f)

val = data.get('validation_results', {})
mem = data.get('memory_results', {})

orig_acc = val.get('direction_accuracy', {}).get('original', 0) * 100
quant_acc = val.get('direction_accuracy', {}).get('quantized', 0) * 100
acc_drop = val.get('direction_accuracy', {}).get('drop', 0) * 100

orig_lat = val.get('latency_ms', {}).get('original', 0)
quant_lat = val.get('latency_ms', {}).get('quantized', 0)
speedup = val.get('latency_ms', {}).get('speedup', 1.0)

mem_gb = mem.get('quantized_gb', 0)

print(f"| $model_name | {orig_acc:.1f}% | {quant_acc:.1f}% | {acc_drop:.1f}%p | {orig_lat:.1f}ms | {quant_lat:.1f}ms | {speedup:.2f}x | {mem_gb:.2f} GB |")
EOF
        fi >> "$COMPARISON_FILE"
    else
        echo "❌ Validation failed for $model_name"
        echo "| $model_name | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR |" >> "$COMPARISON_FILE"
    fi
    
    echo ""
done

# Finalize comparison summary
cat >> "$COMPARISON_FILE" << 'EOF'

---

## Analysis

### Best Model (Accuracy)
- Model with highest quantized accuracy

### Best Model (Speed)
- Model with best speedup ratio

### Best Model (Memory)
- Model with lowest memory footprint

### Recommendation
- For deployment on Jetson 16GB

---

**Generated**: $(date)
EOF

echo ""
echo "================================================================"
echo "✅ All Validations Complete!"
echo "================================================================"
echo ""
echo "📊 Results directory: $RESULTS_DIR"
echo "📄 Comparison summary: $COMPARISON_FILE"
echo ""

# Display comparison
cat "$COMPARISON_FILE"

echo ""
echo "💡 Next steps:"
echo "  1. Review individual results in $RESULTS_DIR"
echo "  2. Select best model for Jetson deployment"
echo "  3. Deploy to Jetson and test"
echo ""
