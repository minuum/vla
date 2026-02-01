#!/bin/bash
# A5000 FastAPI 서버 설정 (Native)
# 실행: bash scripts/setup_api_server.sh

set -e

echo "🚀 A5000 FastAPI 서버 설정 시작"
echo ""

# 1. Python 환경 확인
echo "1. Python 환경 확인..."
python3 --version || { echo "❌ Python3가 설치되지 않았습니다"; exit 1; }
pip3 --version || { echo "❌ pip3가 설치되지 않았습니다"; exit 1; }
echo "   ✅ Python 환경 OK"
echo ""

# 2. 필수 패키지 설치
echo "2. 필수 패키지 설치..."
echo "   설치 중: fastapi, uvicorn, requests, pillow"

pip3 install -q fastapi uvicorn[standard] requests pillow || {
    echo "   ⚠️  일부 패키지 설치 실패, 계속 진행..."
}
echo "   ✅ 패키지 설치 완료"
echo ""

# 3. CUDA & GPU 확인
echo "3. CUDA 및 GPU 확인..."
python3 << EOF
import torch
if torch.cuda.is_available():
    print(f"   ✅ CUDA 사용 가능")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA Version: {torch.version.cuda}")
else:
    print("   ⚠️  CUDA 사용 불가 (CPU 모드)")
EOF
echo ""

# 4. 체크포인트 확인
echo "4. 모델 체크포인트 확인..."
CHECKPOINT="runs/mobile_vla_no_chunk_20251209/checkpoints/epoch=04-val_loss=0.001.ckpt"

if [ -f "$CHECKPOINT" ]; then
    SIZE=$(du -h "$CHECKPOINT" | cut -f1)
    echo "   ✅ Checkpoint found: $SIZE"
    echo "   경로: $CHECKPOINT"
else
    echo "   ⚠️  Best checkpoint not found!"
    echo "   Expected: $CHECKPOINT"
    echo ""
    echo "   대체 체크포인트 찾기..."
    LATEST=$(find runs -name "*.ckpt" -type f 2>/dev/null | head -1)
    if [ -n "$LATEST" ]; then
        echo "   Found: $LATEST"
    else
        echo "   ❌ 체크포인트를 찾을 수 없습니다"
    fi
fi
echo ""

# 5. 방화벽 설정
echo "5. 방화벽 포트 8000 설정..."
if command -v ufw &> /dev/null; then
    sudo ufw allow 8000/tcp 2>/dev/null && echo "   ✅ 포트 8000 열림" || echo "   ⚠️  방화벽 설정 실패 (수동 설정 필요)"
else
    echo "   ⚠️  ufw 없음, 방화벽 수동 설정 필요"
fi
echo ""

# 6. IP 주소 확인
echo "6. 서버 IP 주소:"
echo "   내부 IP:"
hostname -I | awk '{print "     " $1}'

if command -v curl &> /dev/null; then
    EXTERNAL_IP=$(curl -s ifconfig.me 2>/dev/null || echo "확인 실패")
    echo "   외부 IP: $EXTERNAL_IP"
fi
echo ""

# 7. 환경 변수 설정 제안
echo "7. 환경 변수 (선택사항):"
echo "   export VLA_CHECKPOINT_PATH=\"$CHECKPOINT\""
echo ""

# 8. 실행 명령어
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ 설정 완료!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📌 서버 실행 방법:"
echo ""
echo "# 포그라운드 실행 (테스트용)"
echo "python3 Mobile_VLA/inference_server.py"
echo ""
echo "# 백그라운드 실행 (운영용)"
echo "nohup python3 Mobile_VLA/inference_server.py > logs/api_server.log 2>&1 &"
echo ""
echo "# Health check"
IP=$(hostname -I | awk '{print $1}')
echo "curl http://localhost:8000/health"
echo "curl http://${IP}:8000/health"
echo ""
echo "# API 테스트"
echo "python3 scripts/test_inference_api.py"
echo ""
