#!/bin/bash

# 🚀 RoboVLMs Docker 실행 스크립트 - 최소한 설정
# 문제 진단용

set -e

echo "🔍 RoboVLMs Docker 환경 진단 시작"

# 기존 컨테이너 정리
echo "🧹 기존 컨테이너 정리 중..."
docker stop mobile_vla_robovlms_final 2>/dev/null || true
docker rm mobile_vla_robovlms_final 2>/dev/null || true

# 시스템 상태 확인
echo "📊 시스템 상태 확인:"
echo "메모리:"
free -h
echo ""
echo "디스크:"
df -h /
echo ""
echo "Docker 상태:"
docker system df

# 최소한의 설정으로 컨테이너 실행
echo "🚀 최소한 설정으로 컨테이너 실행 중..."
docker run -d \
    --name mobile_vla_robovlms_final \
    --gpus all \
    mobile_vla:robovlms-final \
    tail -f /dev/null

# 컨테이너 상태 확인
echo "📋 컨테이너 상태 확인:"
docker ps -a | grep mobile_vla_robovlms_final

# 로그 확인
echo "📝 컨테이너 로그:"
docker logs mobile_vla_robovlms_final

# 성공하면 접속
if docker ps | grep -q mobile_vla_robovlms_final; then
    echo "✅ 컨테이너가 실행 중입니다. 접속합니다..."
    docker exec -it mobile_vla_robovlms_final bash
else
    echo "❌ 컨테이너가 종료되었습니다."
    echo "📋 종료 로그:"
    docker logs mobile_vla_robovlms_final
fi
