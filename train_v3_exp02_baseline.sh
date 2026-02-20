#!/bin/bash

# V3-EXP-02 Baseline Training Script
# Migration from V2 EXP-17: Window 8, LR 1e-4, Acc 8, No Augmentation
# Model: Kosmos-2 + Classification Head (9 bins)

export CUDA_VISIBLE_DEVICES=0

cd RoboVLMs_upstream

python3 -u main.py ../Mobile_VLA/configs/mobile_vla_v3_exp02_baseline.json 2>&1 | tee ../logs/train_v3_exp02_baseline.log

echo "✅ V3-EXP-02 Baseline Training Started!"
