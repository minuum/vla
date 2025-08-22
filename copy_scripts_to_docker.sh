#!/bin/bash

# =============================================================================
# 📋 Copy Scripts to Docker Container
# Docker 컨테이너에 ROS2 스크립트를 복사하는 스크립트
# =============================================================================

echo "==============================================================================="
echo "📋 Copy Scripts to Docker Container"
echo "==============================================================================="

# 1. 실행 중인 컨테이너 확인
if docker ps | grep -q "infallible_elion"; then
    echo "✅ 실행 중인 컨테이너 발견: infallible_elion"
    
    # 2. 스크립트 복사
    echo "📋 스크립트 복사 중..."
    docker cp setup_ros2_docker.sh infallible_elion:/workspace/vla/
    docker cp run_ros2_docker.sh infallible_elion:/workspace/vla/
    
    # 3. 실행 권한 설정
    echo "🔧 실행 권한 설정 중..."
    docker exec infallible_elion chmod +x /workspace/vla/setup_ros2_docker.sh
    docker exec infallible_elion chmod +x /workspace/vla/run_ros2_docker.sh
    
    echo "✅ 스크립트 복사 완료!"
    echo "📋 컨테이너에서 사용 방법:"
    echo "   docker exec -it infallible_elion bash"
    echo "   cd /workspace/vla"
    echo "   ./setup_ros2_docker.sh"
    echo "   ./run_ros2_docker.sh"
else
    echo "❌ 실행 중인 컨테이너가 없습니다."
    echo "📋 컨테이너를 먼저 시작하세요:"
    echo "   ./run_mobile_vla_system.sh"
fi
