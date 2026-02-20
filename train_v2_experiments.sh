#!/bin/bash

# Experiment V2 Training Script
# 1. EXP-V2-17 (Window 8, Chunk 1, Continuous)
# 2. EXP-V2-12 (Window 6, Chunk 1, Resampler, Continuous)

set -e # Exit immediately if a command exits with a non-zero status.

echo "🚀 Starting V2 Experiment Series (Current Data: basket_dataset_v2)"

# Setup environment if needed
# export CUDA_VISIBLE_DEVICES=0

# --- EXP-V2-17 ---
echo "--------------------------------------------------"
echo "🏃 Running EXP-V2-17 (Window 8, Chunk 1)"
echo "--------------------------------------------------"
cd RoboVLMs_upstream
python3 main.py ../Mobile_VLA/configs/mobile_vla_exp_v2_17.json

# --- EXP-V2-12 ---
echo "--------------------------------------------------"
echo "🏃 Running EXP-V2-12 (Window 6, Chunk 1, Resampler)"
echo "--------------------------------------------------"
python3 main.py ../Mobile_VLA/configs/mobile_vla_exp_v2_12.json

echo "✅ V2 Experiment Series Complete!"
