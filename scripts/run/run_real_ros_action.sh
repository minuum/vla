#!/bin/bash

echo "🚀 실제 ROS_action 시스템 실행 시작!"
echo "=" * 60

# 기존 컨테이너 정리
echo "🧹 기존 컨테이너 정리 중..."
docker stop ros2_camera ros2_inference ros2_control ros2_monitor 2>/dev/null || true
docker rm ros2_camera ros2_inference ros2_control ros2_monitor 2>/dev/null || true

# ROS2 환경 변수 설정
export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_LOCALHOST_ONLY=0

echo "🔧 ROS2 환경 변수 설정:"
echo "   ROS_DOMAIN_ID: $ROS_DOMAIN_ID"
echo "   RMW_IMPLEMENTATION: $RMW_IMPLEMENTATION"
echo "   ROS_LOCALHOST_ONLY: $ROS_LOCALHOST_ONLY"

# 컨테이너 실행
echo "📦 컨테이너 실행 중..."

# 카메라 컨테이너
echo "📷 카메라 컨테이너 시작..."
docker run -d --name ros2_camera \
    --net=host \
    -e ROS_DOMAIN_ID=$ROS_DOMAIN_ID \
    -e RMW_IMPLEMENTATION=$RMW_IMPLEMENTATION \
    -e ROS_LOCALHOST_ONLY=$ROS_LOCALHOST_ONLY \
    -v $(pwd):/workspace/vla \
    mobile_vla:pytorch-2.3.0-cuda \
    sleep infinity

# 추론 컨테이너
echo "🧠 추론 컨테이너 시작..."
docker run -d --name ros2_inference \
    --net=host \
    -e ROS_DOMAIN_ID=$ROS_DOMAIN_ID \
    -e RMW_IMPLEMENTATION=$RMW_IMPLEMENTATION \
    -e ROS_LOCALHOST_ONLY=$ROS_LOCALHOST_ONLY \
    -v $(pwd):/workspace/vla \
    mobile_vla:pytorch-2.3.0-cuda \
    sleep infinity

# 제어 컨테이너
echo "🤖 제어 컨테이너 시작..."
docker run -d --name ros2_control \
    --net=host \
    -e ROS_DOMAIN_ID=$ROS_DOMAIN_ID \
    -e RMW_IMPLEMENTATION=$RMW_IMPLEMENTATION \
    -e ROS_LOCALHOST_ONLY=$ROS_LOCALHOST_ONLY \
    -v $(pwd):/workspace/vla \
    mobile_vla:pytorch-2.3.0-cuda \
    sleep infinity

# 모니터링 컨테이너
echo "📊 모니터링 컨테이너 시작..."
docker run -d --name ros2_monitor \
    --net=host \
    -e ROS_DOMAIN_ID=$ROS_DOMAIN_ID \
    -e RMW_IMPLEMENTATION=$RMW_IMPLEMENTATION \
    -e ROS_LOCALHOST_ONLY=$ROS_LOCALHOST_ONLY \
    -v $(pwd):/workspace/vla \
    mobile_vla:pytorch-2.3.0-cuda \
    sleep infinity

# 컨테이너 상태 확인
echo "📋 컨테이너 상태 확인..."
docker ps | grep ros2

# 잠시 대기
echo "⏳ 컨테이너 안정화 대기 중..."
sleep 3

# 실제 ROS 노드 실행
echo "🎯 실제 ROS 노드 실행 중..."

# 카메라 노드 실행
echo "📷 카메라 노드 시작..."
docker exec -d ros2_camera bash -c "
    cd /workspace/vla
    source /opt/ros/humble/setup.bash
    cd ROS_action
    colcon build --packages-select camera_pub
    source install/setup.bash
    ros2 run camera_pub camera_publisher_continuous
"

# 추론 노드 실행
echo "🧠 추론 노드 시작..."
docker exec -d ros2_inference bash -c "
    cd /workspace/vla
    source /opt/ros/humble/setup.bash
    cd ROS_action
    colcon build --packages-select mobile_vla_package
    source install/setup.bash
    ros2 run mobile_vla_package robovlms_inference
"

# 제어 노드 실행
echo "🤖 제어 노드 시작..."
docker exec -d ros2_control bash -c "
    cd /workspace/vla
    source /opt/ros/humble/setup.bash
    cd ROS_action
    colcon build --packages-select mobile_vla_package
    source install/setup.bash
    ros2 run mobile_vla_package simple_robot_mover
"

# 모니터링 노드 실행
echo "📊 모니터링 노드 시작..."
docker exec -d ros2_monitor bash -c "
    cd /workspace/vla
    source /opt/ros/humble/setup.bash
    cd ROS_action
    colcon build --packages-select mobile_vla_package
    source install/setup.bash
    ros2 run mobile_vla_package robovlms_monitor
"

# 잠시 대기
echo "⏳ 노드 시작 대기 중..."
sleep 5

# 시스템 상태 확인
echo "🔍 시스템 상태 확인..."
echo "📋 실행 중인 노드:"
docker exec ros2_camera bash -c "source /opt/ros/humble/setup.bash && ros2 node list" 2>/dev/null || echo "노드 목록 확인 실패"

echo "📋 활성 토픽:"
docker exec ros2_camera bash -c "source /opt/ros/humble/setup.bash && ros2 topic list" 2>/dev/null || echo "토픽 목록 확인 실패"

echo "📋 서비스 목록:"
docker exec ros2_camera bash -c "source /opt/ros/humble/setup.bash && ros2 service list" 2>/dev/null || echo "서비스 목록 확인 실패"

echo ""
echo "✅ 실제 ROS_action 시스템 실행 완료!"
echo "📋 사용 가능한 명령어:"
echo "   📷 카메라 로그: docker logs -f ros2_camera"
echo "   🧠 추론 로그: docker logs -f ros2_inference"
echo "   🤖 제어 로그: docker logs -f ros2_control"
echo "   📊 모니터링 로그: docker logs -f ros2_monitor"
echo "   🛑 시스템 중지: docker stop ros2_camera ros2_inference ros2_control ros2_monitor"
echo "   🧹 시스템 정리: docker rm ros2_camera ros2_inference ros2_control ros2_monitor"
echo ""
echo "🎮 제어 노드에서 WASD 키로 로봇을 조작할 수 있습니다!"
echo "📊 모니터링 노드에서 실시간 시스템 상태를 확인할 수 있습니다!"
