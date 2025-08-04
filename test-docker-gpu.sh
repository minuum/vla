#!/bin/bash
echo "🧪 Docker GPU 지원 테스트..."
docker run --rm --runtime=nvidia --gpus all \
  nvcr.io/nvidia/l4t-base:r36.4.0 \
  python3 -c "
import platform
print(f'🖥️  Platform: {platform.platform()}')
print(f'🏗️  Architecture: {platform.machine()}')

try:
    import torch
    print(f'🔥 PyTorch: {torch.__version__}')
    print(f'🎯 CUDA Available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'📟 CUDA Device: {torch.cuda.get_device_name(0)}')
        print(f'💾 CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
except ImportError:
    print('⚠️  PyTorch not available in base image')

print('✅ Docker GPU 테스트 완료!')
"
