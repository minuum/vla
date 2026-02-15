#!/bin/bash
# 네트워크 연결 테스트 스크립트
# 로봇 서버에서 실행: A5000 서버 연결 확인
# 
# 사용법: bash scripts/test_network.sh <a5000_ip>

set -e

BILLY_IP="${1:-localhost}"
PORT="8000"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔍 A5000 서버 연결 테스트"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "   서버: ${BILLY_IP}:${PORT}"
echo ""

# 1. Ping 테스트
echo "1️⃣  Ping 테스트..."
if ping -c 3 -W 2 "${BILLY_IP}" > /dev/null 2>&1; then
    echo "   ✅ Ping 성공"
else
    echo "   ❌ Ping 실패 - 네트워크 연결 확인 필요"
    echo ""
    echo "   해결 방법:"
    echo "   - 네트워크 케이블 확인"
    echo "   - 같은 네트워크인지 확인"
    echo "   - IP 주소 확인: ${BILLY_IP}"
    exit 1
fi

# 2. 포트 테스트
echo ""
echo "2️⃣  포트 ${PORT} 테스트..."

if command -v nc &> /dev/null; then
    if nc -zv "${BILLY_IP}" "${PORT}" 2>&1 | grep -q "succeeded\|open"; then
        echo "   ✅ 포트 열림"
    else
        echo "   ❌ 포트 닫힘"
        echo ""
        echo "   해결 방법:"
        echo "   - A5000에서 API 서버 실행 확인"
        echo "   - 방화벽 설정: sudo ufw allow 8000/tcp"
        exit 1
    fi
else
    echo "   ⚠️  nc 명령어 없음, 건너뛰기"
fi

# 3. API Health Check
echo ""
echo "3️⃣  API Health Check..."

if command -v curl &> /dev/null; then
    RESPONSE=$(curl -s -w "\n%{http_code}" "http://${BILLY_IP}:${PORT}/health" 2>/dev/null)
    HTTP_CODE=$(echo "$RESPONSE" | tail -1)
    BODY=$(echo "$RESPONSE" | head -n -1)
    
    if [ "$HTTP_CODE" = "200" ]; then
        echo "   ✅ API 정상 (HTTP 200)"
        echo ""
        echo "   응답:"
        echo "$BODY" | python3 -m json.tool 2>/dev/null || echo "$BODY"
    else
        echo "   ❌ API 응답 이상 (HTTP ${HTTP_CODE})"
        echo "   응답: $BODY"
        exit 1
    fi
else
    echo "   ⚠️  curl 명령어 없음, 건너뛰기"
fi

# 4. Latency 측정
echo ""
echo "4️⃣  Latency 측정 (5회)..."

if command -v curl &> /dev/null; then
    TOTAL=0
    COUNT=0
    
    for i in {1..5}; do
        TIME=$(curl -w "%{time_total}" -o /dev/null -s "http://${BILLY_IP}:${PORT}/health" 2>/dev/null || echo "0")
        TIME_MS=$(echo "$TIME * 1000" | bc 2>/dev/null || echo "0")
        
        if [ "$TIME_MS" != "0" ]; then
            echo "   시도 $i: ${TIME_MS} ms"
            TOTAL=$(echo "$TOTAL + $TIME_MS" | bc)
            COUNT=$((COUNT + 1))
        fi
    done
    
    if [ $COUNT -gt 0 ]; then
        AVG=$(echo "scale=1; $TOTAL / $COUNT" | bc)
        echo ""
        echo "   평균 Latency: ${AVG} ms"
        
        # 경고 임계값
        THRESHOLD=100
        if (( $(echo "$AVG > $THRESHOLD" | bc -l) )); then
            echo "   ⚠️  Latency가 높습니다 (>${THRESHOLD}ms)"
            echo "   실시간 제어에 영향이 있을 수 있습니다"
        else
            echo "   ✅ Latency 양호 (<${THRESHOLD}ms)"
        fi
    fi
fi

# 5. Test 엔드포인트 확인
echo ""
echo "5️⃣  Test 엔드포인트 확인..."

if command -v curl &> /dev/null; then
    RESPONSE=$(curl -s "http://${BILLY_IP}:${PORT}/test" 2>/dev/null)
    
    if echo "$RESPONSE" | grep -q "action"; then
        echo "   ✅ Test 엔드포인트 정상"
        
        # Action 값 추출
        ACTION=$(echo "$RESPONSE" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('action', []))" 2>/dev/null || echo "")
        if [ -n "$ACTION" ]; then
            echo "   Sample action: $ACTION"
        fi
    else
        echo "   ⚠️  Test 엔드포인트 응답 확인 불가"
    fi
fi

# 결과 요약
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ 모든 테스트 통과!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📌 다음 단계:"
echo ""
echo "# 환경 변수 설정"
echo "export VLA_API_SERVER=\"http://${BILLY_IP}:${PORT}\""
echo ""
echo "# Python 클라이언트 실행"
echo "python3 vla_api_client.py --test"
echo ""
