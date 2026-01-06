#!/bin/bash
# Dual Strategy API Server Quick Start Script
# Step 1: API 서버 실제 테스트

set -e  # Exit on error

echo "════════════════════════════════════════════════════════════════"
echo "🚀 Step 1: Dual Strategy API Server Test"
echo "════════════════════════════════════════════════════════════════"

# 1. Check files
echo ""
echo "📂 Checking required files..."

CHECKPOINT="runs/mobile_vla_no_chunk_20251209/kosmos/mobile_vla_finetune/2025-12-18/mobile_vla_left_chunk10_20251218/epoch_epoch=09-val_loss=val_loss=0.010.ckpt"
CONFIG="Mobile_VLA/configs/mobile_vla_left_chunk10_20251218.json"

if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ Checkpoint not found: $CHECKPOINT"
    exit 1
fi
echo "✅ Checkpoint: $CHECKPOINT"

if [ ! -f "$CONFIG" ]; then
    echo "❌ Config not found: $CONFIG"
    exit 1
fi
echo "✅ Config: $CONFIG"

if [ ! -f "Mobile_VLA/inference_server_dual.py" ]; then
    echo "❌ Server script not found"
    exit 1
fi
echo "✅ Server: Mobile_VLA/inference_server_dual.py"

if [ ! -f "scripts/test_dual_strategy.py" ]; then
    echo "❌ Test script not found"
    exit 1
fi
echo "✅ Test script: scripts/test_dual_strategy.py"

# 2. Set environment variables
echo ""
echo "🔧 Setting environment variables..."

export VLA_API_KEY="mobile-vla-test-$(date +%s)"
export VLA_CHECKPOINT_PATH="$CHECKPOINT"
export VLA_CONFIG_PATH="$CONFIG"

echo "✅ VLA_API_KEY=$VLA_API_KEY"
echo "✅ VLA_CHECKPOINT_PATH=$VLA_CHECKPOINT_PATH"
echo "✅ VLA_CONFIG_PATH=$VLA_CONFIG_PATH"

# 3. Save API key for test script
echo ""
echo "💾 Saving API key for test script..."
echo "export VLA_API_KEY=\"$VLA_API_KEY\"" > /tmp/vla_api_key.sh
chmod +x /tmp/vla_api_key.sh
echo "✅ Saved to /tmp/vla_api_key.sh"

# 4. Instructions
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "📝 Next Steps:"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Terminal 1 (API Server):"
echo "  source /tmp/vla_api_key.sh"
echo "  python Mobile_VLA/inference_server_dual.py"
echo ""
echo "Terminal 2 (Test):"
echo "  source /tmp/vla_api_key.sh"
echo "  python scripts/test_dual_strategy.py"
echo ""
echo "Or run automated test:"
echo "  bash scripts/run_dual_strategy_test.sh"
echo ""
echo "════════════════════════════════════════════════════════════════"

# 5. Ask if user wants to start server now
echo ""
read -p "Start API server now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "🚀 Starting API server..."
    echo ""
    python Mobile_VLA/inference_server_dual.py
fi
