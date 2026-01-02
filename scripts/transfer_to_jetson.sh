#!/bin/bash
# Transfer Checkpoint to Jetson
# Usage: ./transfer_to_jetson.sh jetson@jetson-ip

set -e

JETSON_TARGET=${1:-"soda@linnaeus"}  # Tailscale: soda@linnaeus (100.85.118.58)
VLA_DIR="/home/soda/25-1kp/vla"

echo "========================================"
echo "Transfer Checkpoint to Jetson"
echo "========================================"
echo "Target: $JETSON_TARGET"
echo ""

cd $VLA_DIR

# 1. Checkpoint 준비
CHECKPOINT="runs/mobile_vla_no_chunk_20251209/kosmos/mobile_vla_finetune/2025-12-17/mobile_vla_chunk5_20251217/epoch_epoch=06-val_loss=val_loss=0.067.ckpt"
CONFIG="Mobile_VLA/configs/mobile_vla_chunk5_20251217.json"

if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ Checkpoint not found: $CHECKPOINT"
    exit 1
fi

echo "✅ Checkpoint found:"
ls -lh "$CHECKPOINT"
echo ""

# 2. 압축
echo "Compressing checkpoint (this may take a few minutes)..."
CHECKPOINT_TAR="/tmp/vla_checkpoint_$(date +%Y%m%d).tar.gz"

tar -czf "$CHECKPOINT_TAR" \
    --transform 's|.*/||' \
    "$CHECKPOINT" \
    "$CONFIG"

echo "✅ Compressed to: $CHECKPOINT_TAR"
ls -lh "$CHECKPOINT_TAR"
echo ""

# 3. Pretrained model 체크
PRETRAINED_DIR=".vlms/kosmos-2-patch14-224"
if [ -d "$PRETRAINED_DIR" ]; then
    echo "Compressing Kosmos-2 pretrained model..."
    PRETRAINED_TAR="/tmp/vla_kosmos2_$(date +%Y%m%d).tar.gz"
    tar -czf "$PRETRAINED_TAR" "$PRETRAINED_DIR"
    echo "✅ Compressed to: $PRETRAINED_TAR"
    ls -lh "$PRETRAINED_TAR"
    echo ""
else
    echo "⚠️  Pretrained model not found (will download on Jetson)"
    PRETRAINED_TAR=""
fi

# 4. 전송
echo "Transferring to Jetson..."
echo "Target: $JETSON_TARGET"
echo ""

# Checkpoint 전송
echo "1. Transferring checkpoint..."
if scp "$CHECKPOINT_TAR" "$JETSON_TARGET:/tmp/"; then
    echo "   ✅ Checkpoint transferred"
else
    echo "   ❌ Transfer failed"
    exit 1
fi

# Pretrained model 전송 (있으면)
if [ -n "$PRETRAINED_TAR" ]; then
    echo "2. Transferring Kosmos-2..."
    if scp "$PRETRAINED_TAR" "$JETSON_TARGET:/tmp/"; then
        echo "   ✅ Kosmos-2 transferred"
    else
        echo "   ⚠️  Kosmos-2 transfer failed (can download later)"
    fi
fi

# 5. Cleanup
echo ""
echo "Cleaning up local temp files..."
rm -f "$CHECKPOINT_TAR" "$PRETRAINED_TAR"
echo "✅ Cleanup done"

# 6. Instructions
echo ""
echo "========================================"
echo "✅ Transfer Complete!"
echo "========================================"
echo ""
echo "Next steps on Jetson:"
echo ""
echo "1. Extract checkpoint:"
echo "   cd ~/vla"
echo "   tar -xzf /tmp/vla_checkpoint_*.tar.gz --strip-components=5 -C runs/mobile_vla_no_chunk_20251209/kosmos/mobile_vla_finetune/2025-12-17/mobile_vla_chunk5_20251217/"
echo ""
if [ -n "$PRETRAINED_TAR" ]; then
echo "2. Extract Kosmos-2:"
echo "   cd ~/vla"
echo "   tar -xzf /tmp/vla_kosmos2_*.tar.gz"
echo ""
fi
echo "3. Test:"
echo "   source secrets.sh"
echo "   python3 -m uvicorn Mobile_VLA.inference_server:app --host 0.0.0.0 --port 8000"
echo ""
