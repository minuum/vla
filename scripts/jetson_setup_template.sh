#!/bin/bash
# Jetson에서 실행할 설정 파일 (템플릿)
# 이 파일을 Jetson으로 복사한 후, 아래 TODO 부분을 실제 값으로 채우세요.

echo "=================================================="
echo "🤖 Jetson Setup for Billy API Connection"
echo "=================================================="
echo ""
echo "이 템플릿을 Jetson에 복사한 후, secrets.sh를 만드세요:"
echo ""
cat << 'EOF'
# Jetson에서 실행: nano secrets.sh
# 아래 내용을 secrets.sh에 저장하세요:

# Billy 서버 주소 (Tailscale IP)
export BILLY_IP="100.86.152.29"
export BILLY_URL="http://100.86.152.29:8000"

# API Key (Billy가 검증에 사용하는 키)
export VLA_API_KEY="YOUR_API_KEY_HERE"  # ← Billy 관리자에게 받으세요

EOF
echo ""
echo "=================================================="
echo "📋 Jetson에서 실행할 명령어:"
echo "=================================================="
echo ""
echo "1. secrets.sh 생성 후 위 내용을 저장"
echo "2. source secrets.sh"
echo "3. curl -H \"X-API-Key: \$VLA_API_KEY\" \$BILLY_URL/health"
echo ""
echo "✅ 테스트에 성공하면 ROS2 client를 실행하세요."
echo "=================================================="
