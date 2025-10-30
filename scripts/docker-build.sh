#!/bin/bash
echo "🔨 Mobile VLA Jetson Docker 이미지 빌드 중..."
docker-compose -f docker-compose.jetson.yml build --no-cache mobile-vla
echo "✅ 빌드 완료!"
