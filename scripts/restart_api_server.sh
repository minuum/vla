#!/bin/bash
# API 서버 재시작 스크립트 (Left Chunk10 Best Model)

echo "🔄 API 서버 재시작 중..."

# 기존 서버 종료
echo "1. 기존 API 서버 확인 및 종료..."
pkill -f "inference_server.py" || echo "  (실행 중인 서버 없음)"
sleep 2

# Best Model 경로 설정
export VLA_CHECKPOINT_PATH="runs/mobile_vla_no_chunk_20251209/kosmos/mobile_vla_finetune/2025-12-18/mobile_vla_left_chunk10_20251218/epoch_epoch=09-val_loss=val_loss=0.010.ckpt"
export VLA_CONFIG_PATH="Mobile_VLA/configs/mobile_vla_left_chunk10_20251218.json"
export VLA_API_KEY="${VLA_API_KEY:-default-api-key-change-me}"

echo "2. Best Model 설정..."
echo "   Checkpoint: $VLA_CHECKPOINT_PATH"
echo "   Config: $VLA_CONFIG_PATH"

# 로그 디렉토리 생성
mkdir -p logs

# API 서버 시작
echo "3. API 서버 시작..."
nohup python3 Mobile_VLA/inference_server.py > logs/api_server_left_chunk10_$(date +%Y%m%d_%H%M%S).log 2>&1 &

SERVER_PID=$!
echo "   PID: $SERVER_PID"

# 시작 대기
echo "4. 서버 시작 대기 중..."
sleep 5

# Health check
echo "5. Health check..."
if curl -s http://localhost:8000/health | grep -q "healthy"; then
    echo "   ✅ API 서버 정상 실행 중"
    echo ""
    echo "📊 서버 정보:"
    curl -s http://localhost:8000/health | python3 -m json.tool
else
    echo "   ❌ API 서버 시작 실패"
    echo "   로그 확인: tail -f logs/api_server_left_chunk10_*.log"
    exit 1
fi

echo ""
echo "✅ API 서버 재시작 완료!"
echo "   PID: $SERVER_PID"
echo "   로그: logs/api_server_left_chunk10_*.log"
