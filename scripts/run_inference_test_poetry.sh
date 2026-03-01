#!/bin/bash
# VLA 추론 시스템 단계별 테스트 (Poetry 환경)

set -e

# 환경 설정
export PATH="/home/billy/.local/bin:$PATH"
export POETRY_PYTHON=/home/billy/.cache/pypoetry/virtualenvs/robovlms-ASlHafON-py3.10/bin/python
export VLA_CHECKPOINT_PATH="/home/billy/25-1kp/vla/RoboVLMs_upstream/runs/mobile_vla_kosmos2_right_only_20251207/kosmos/mobile_vla_finetune/2025-12-07/mobile_vla_kosmos2_right_only_20251207/last.ckpt"

echo "🔧 환경 설정 완료"
echo "📌 Poetry Python: $POETRY_PYTHON"
echo "📌 체크포인트: $VLA_CHECKPOINT_PATH"
echo ""

# Python 환경 확인
echo "🐍 Python 버전 확인..."
$POETRY_PYTHON --version

echo ""
echo "📦 주요 패키지 확인..."
$POETRY_PYTHON -c "import torch; print('✅ torch:', torch.__version__)" 2>/dev/null || echo "❌ torch 없음"
$POETRY_PYTHON -c "import lightning; print('✅ lightning:', lightning.__version__)" 2>/dev/null || echo "❌ lightning 없음"
$POETRY_PYTHON -c "import transformers; print('✅ transformers:', transformers.__version__)" 2>/dev/null || echo "❌ transformers 없음"

echo ""
echo "🚀 추론 테스트 시작..."
echo "="*60

cd /home/billy/25-1kp/vla
$POETRY_PYTHON test_inference_stepbystep.py
