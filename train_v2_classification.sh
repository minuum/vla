#!/bin/bash

# V2 Classification Training Script (9-classes)
# 목표: 모호한 연속값(Regression) 대신 확실한 키 입력(Classification)으로 학습

set -e

echo "🚀 Starting V2 Classification Training (9-classes)"
echo "Target: Omniwheel Discrete Movement Logic"

cd RoboVLMs_upstream
python3 main.py ../Mobile_VLA/configs/mobile_vla_v2_classification_9cls.json

echo "✅ Classification Training Complete!"
