#!/bin/bash

# =============================================================================
# EXP-V3-01: V3 Classification Training with Color Jitter + Random Crop
# 목표: 시각적 일반화 개선 (조명 변화 / 카메라 각도 변화 대응)
# 변경점 vs V2:
#   - Color Jitter (brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1)
#   - Random Crop (RandomResizedCrop, scale=0.8~1.0)
#   - LR: 1e-4 → 5e-5 (더 보수적인 학습, 과적합 방지)
#   - max_epochs: 10 → 15 (augmentation으로 유효 데이터가 늘어나므로)
# =============================================================================

set -e

echo "🚀 Starting V3-EXP-01: Classification + Color Jitter + Random Crop"
echo "============================================================"
echo "Config: Mobile_VLA/configs/mobile_vla_v3_exp01_aug.json"
echo "Dataset: basket_dataset_v2 (528 episodes)"
echo "Augmentation: Color Jitter + Random ResizedCrop"
echo "LR: 5e-5 (conservative)"
echo "Max Epochs: 15"
echo "============================================================"

cd RoboVLMs_upstream
python3 main.py ../Mobile_VLA/configs/mobile_vla_v3_exp01_aug.json

echo "✅ V3-EXP-01 Training Complete!"
