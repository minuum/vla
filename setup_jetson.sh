#!/bin/bash
# Mobile VLA API Server - Jetson Setup Script
# For Jetson Orin / Xavier (ARM64 architecture)

set -e

echo "========================================"
echo "Mobile VLA API Server - Jetson Setup"
echo "========================================"
echo ""

# Check if running on Jetson
if [ -f /etc/nv_tegra_release ]; then
    echo "✅ Jetson device detected"
    cat /etc/nv_tegra_release
else
    echo "⚠️  Warning: Not running on Jetson"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "1. Checking Python version..."
python3 --version
if [ $? -ne 0 ]; then
    echo "❌ Python 3 not found"
    exit 1
fi

# Check Python version >= 3.10
PY_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "   Python version: $PY_VERSION"

echo ""
echo "2. Checking CUDA..."
if command -v nvcc &> /dev/null; then
    nvcc --version | grep "release"
    echo "   ✅ CUDA found"
else
    echo "   ⚠️  CUDA not found - BitsAndBytes may not work"
fi

echo ""
echo "3. Installing dependencies..."
echo "   This may take 10-15 minutes on Jetson..."

# Upgrade pip
python3 -m pip install --upgrade pip

# Install PyTorch (Jetson에 맞는 버전)
echo ""
echo "   Installing PyTorch for Jetson..."
# Jetson에는 이미 PyTorch가 설치되어 있을 수 있음
if python3 -c "import torch" 2>/dev/null; then
    echo "   ✅ PyTorch already installed"
    python3 -c "import torch; print(f'   PyTorch version: {torch.__version__}')"
else
    echo "   Installing PyTorch..."
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
fi

# Install other dependencies
echo ""
echo "   Installing other dependencies..."
pip3 install -r requirements-inference.txt

echo ""
echo "4. Verifying installation..."

# Check critical packages
python3 << EOF
import sys
required_packages = {
    'torch': 'PyTorch',
    'transformers': 'Transformers',
    'bitsandbytes': 'BitsAndBytes',
    'accelerate': 'Accelerate',
    'fastapi': 'FastAPI',
    'uvicorn': 'Uvicorn',
    'PIL': 'Pillow',
}

missing = []
for module, name in required_packages.items():
    try:
        __import__(module)
        print(f'   ✅ {name}')
    except ImportError:
        print(f'   ❌ {name} - MISSING')
        missing.append(name)

if missing:
    print(f'\n❌ Missing packages: {", ".join(missing)}')
    sys.exit(1)
else:
    print('\n✅ All packages installed successfully!')
EOF

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Installation verification failed"
    exit 1
fi

echo ""
echo "5. Checking model files..."
MODEL_PATH="runs/mobile_vla_no_chunk_20251209/kosmos/mobile_vla_finetune/2025-12-17/mobile_vla_chunk5_20251217"
CHECKPOINT="$MODEL_PATH/epoch_epoch=06-val_loss=val_loss=0.067.ckpt"
CONFIG="Mobile_VLA/configs/mobile_vla_chunk5_20251217.json"

if [ -f "$CHECKPOINT" ]; then
    echo "   ✅ Checkpoint found"
    ls -lh "$CHECKPOINT"
else
    echo "   ❌ Checkpoint not found: $CHECKPOINT"
    echo "   You need to sync the checkpoint from Billy server"
    echo "   Run: ./scripts/sync/pull_checkpoint_from_billy.sh"
fi

if [ -f "$CONFIG" ]; then
    echo "   ✅ Config found"
else
    echo "   ❌ Config not found: $CONFIG"
fi

echo ""
echo "6. Setting up API Key..."
if [ -z "$VLA_API_KEY" ]; then
    # Generate API key
    API_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
    echo "export VLA_API_KEY=\"$API_KEY\"" >> secrets.sh
    echo "   Generated API Key (saved to secrets.sh):"
    echo "   $API_KEY"
    echo ""
    echo "   To use: source secrets.sh"
else
    echo "   ✅ VLA_API_KEY already set"
fi

echo ""
echo "7. Creating log directory..."
mkdir -p logs
echo "   ✅ logs/ directory created"

echo ""
echo "========================================"
echo "✅ Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Source API key:"
echo "   source secrets.sh"
echo ""
echo "2. Start server:"
echo "   python3 -m uvicorn Mobile_VLA.inference_server:app --host 0.0.0.0 --port 8000"
echo ""
echo "3. Test (in another terminal):"
echo "   curl http://localhost:8000/health"
echo ""
echo "For background mode:"
echo "   nohup python3 -m uvicorn Mobile_VLA.inference_server:app \\"
echo "     --host 0.0.0.0 --port 8000 > logs/api_server.log 2>&1 &"
echo ""
echo "See QUICKSTART.md for more details"
echo ""
