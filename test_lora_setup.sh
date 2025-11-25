#!/bin/bash
# LoRA Fine-tuning 설정 테스트

echo "========================================="
echo "LoRA Fine-tuning 설정 테스트"
echo "========================================="
echo ""

# 1. 파일 존재 확인
echo "📁 파일 확인..."
files=(
    "Mobile_VLA/configs/finetune_mobile_vla_lora_20251106.json"
    "Mobile_VLA/src/data/mobile_vla_h5_dataset.py"
    "Mobile_VLA/src/training/finetune_lora_20251106.py"
    "Mobile_VLA/scripts/run_lora_finetune_20251106.sh"
    "Mobile_VLA/scripts/test_dataset_20251106.py"
)

for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✅ $file"
    else
        echo "  ❌ $file (없음)"
    fi
done
echo ""

# 2. 데이터셋 확인
echo "📊 데이터셋 확인..."
episode_count=$(ls -1 ROS_action/mobile_vla_dataset/episode_20251106_*.h5 2>/dev/null | wc -l)
echo "  20251106 에피소드: ${episode_count}개"
echo ""

# 3. Python 패키지 확인
echo "🐍 Python 패키지 확인..."
python3 -c "
import sys
packages = ['torch', 'transformers', 'peft', 'h5py', 'numpy', 'PIL', 'cv2']
for pkg in packages:
    try:
        __import__(pkg)
        print(f'  ✅ {pkg}')
    except ImportError:
        print(f'  ❌ {pkg} (설치 필요)')
"
echo ""

# 4. CUDA 확인
echo "🔧 CUDA 확인..."
python3 -c "import torch; print(f'  CUDA Available: {torch.cuda.is_available()}'); print(f'  CUDA Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo ""

echo "========================================="
echo "✅ 설정 테스트 완료"
echo "========================================="
echo ""
echo "다음 단계:"
echo "  1. python3 Mobile_VLA/scripts/test_dataset_20251106.py"
echo "  2. bash Mobile_VLA/scripts/run_lora_finetune_20251106.sh"
