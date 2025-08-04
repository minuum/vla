#!/bin/bash
echo "🛑 Mobile VLA Jetson Docker 컨테이너 중지 중..."
docker-compose -f docker-compose.jetson.yml down
echo "✅ 컨테이너 중지 완료!"
