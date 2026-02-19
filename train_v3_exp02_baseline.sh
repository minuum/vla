#!/bin/bash

# V3-EXP-02 Baseline Training Script
# Migration from V2 EXP-17: Window 8, LR 1e-4, Acc 8, No Augmentation
# Model: Kosmos-2 + Classification Head (9 bins)

export CUDA_VISIBLE_DEVICES=0

cd RoboVLMs_upstream

python3 main.py ../Mobile_VLA/configs/mobile_vla_v3_exp02_baseline.json \
    --trainer.max_epochs 10 \
    --learning_rate 1e-4 \
    --trainer.accumulate_grad_batches 8

echo "✅ V3-EXP-02 Baseline Training Started!"
