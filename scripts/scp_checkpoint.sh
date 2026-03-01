#!/bin/bash

# V3-EXP-01 Epoch 3 Checkpoint Transfer Script
# Path to Best Checkpoint (Val Loss: 0.455)
CKPT_PATH="/home/billy/25-1kp/vla/RoboVLMs_upstream/runs/v3_classification/kosmos/mobile_vla_v3_classification/2026-02-19/v3-exp01-aug/epoch_epoch=03-val_loss=val_loss=0.455.ckpt"

# Remote Destination
REMOTE_USER="soda"
REMOTE_IP="100.85.118.58"
REMOTE_PATH="/"

echo "🚀 Transferring V3-EXP-01 Epoch 3 checkpoint to $REMOTE_USER@$REMOTE_IP:$REMOTE_PATH..."

scp "$CKPT_PATH" "$REMOTE_USER@$REMOTE_IP:$REMOTE_PATH"

if [ $? -eq 0 ]; then
    echo "✅ Transfer Complete!"
else
    echo "❌ Transfer Failed!"
fi
