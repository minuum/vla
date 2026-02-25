#!/bin/bash

# V3-EXP-04 LoRA Training Script
# Added: LoRA (Low-Rank Adaptation) on Kosmos-2 backbone with Class-Weighted Loss (FORWARD=0.2, others=5.0)
# Added: Train Text Embedding (Language Condition Tuning) to improve instruction grounding
# Model: Kosmos-2 + Classification Head (9 bins) + LoRA
# Epochs: 10 fixed + Early Stopping

export CUDA_VISIBLE_DEVICES=0

cd RoboVLMs_upstream

python3 -u main.py ../Mobile_VLA/configs/mobile_vla_v3_exp04_lora.json 2>&1 | tee ../logs/train_v3_exp04_lora.log

echo "✅ V3-EXP-04 LoRA Weighted Training Started!"
