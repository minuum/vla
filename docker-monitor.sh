#!/bin/bash
echo "📊 Mobile VLA 모니터링 서비스 시작 중..."
docker-compose -f docker-compose.jetson.yml --profile monitoring up -d
echo "✅ 모니터링 서비스 시작 완료!"
echo "📊 모니터링 로그 확인: docker logs -f mobile_vla_monitoring"
