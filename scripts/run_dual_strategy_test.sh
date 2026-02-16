#!/bin/bash
# Automated Dual Strategy API Test
# Runs server in background and executes test

set -e

echo "════════════════════════════════════════════════════════════════"
echo "🧪 Automated Dual Strategy API Test"
echo "════════════════════════════════════════════════════════════════"

# Setup
CHECKPOINT="runs/mobile_vla_no_chunk_20251209/kosmos/mobile_vla_finetune/2025-12-18/mobile_vla_left_chunk10_20251218/epoch_epoch=09-val_loss=val_loss=0.010.ckpt"
CONFIG="Mobile_VLA/configs/mobile_vla_left_chunk10_20251218.json"
API_KEY="mobile-vla-test-$(date +%s)"

export VLA_API_KEY="$API_KEY"
export VLA_CHECKPOINT_PATH="$CHECKPOINT"
export VLA_CONFIG_PATH="$CONFIG"

echo "✅ Environment configured"
echo ""

# Start server in background
echo "🚀 Starting API server in background..."
python3 Mobile_VLA/inference_server_dual.py > /tmp/vla_server.log 2>&1 &
SERVER_PID=$!

echo "✅ Server PID: $SERVER_PID"
echo "📄 Server log: /tmp/vla_server.log"

# Wait for server to start
echo "⏳ Waiting for server to initialize (15s)..."
sleep 15

# Check if server is running
if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "❌ Server failed to start!"
    echo "Last 20 lines of log:"
    tail -20 /tmp/vla_server.log
    exit 1
fi

echo "✅ Server is running"
echo ""

# Run test
echo "🧪 Running test suite..."
echo ""
python3 scripts/test_dual_strategy.py

TEST_EXIT=$?

# Cleanup
echo ""
echo "🧹 Cleaning up..."
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true
echo "✅ Server stopped"

# Results
echo ""
echo "════════════════════════════════════════════════════════════════"
if [ $TEST_EXIT -eq 0 ]; then
    echo "✅ All tests PASSED!"
else
    echo "❌ Tests FAILED (exit code: $TEST_EXIT)"
fi
echo "════════════════════════════════════════════════════════════════"

exit $TEST_EXIT
