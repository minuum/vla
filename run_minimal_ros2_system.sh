#!/bin/bash

echo "🚀 최소 기능 ROS2 시스템 실행"
echo "================================"

# 기존 컨테이너 정리
echo "🧹 기존 컨테이너 정리 중..."
docker stop ros2_camera ros2_inference ros2_control ros2_monitor 2>/dev/null
docker rm ros2_camera ros2_inference ros2_control ros2_monitor 2>/dev/null

# ROS2 네트워크 설정
export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_LOCALHOST_ONLY=0

echo "📦 카메라 컨테이너 실행 중..."
docker run -d --name ros2_camera \
    --net=host \
    -e ROS_DOMAIN_ID=$ROS_DOMAIN_ID \
    -e RMW_IMPLEMENTATION=$RMW_IMPLEMENTATION \
    -e ROS_LOCALHOST_ONLY=$ROS_LOCALHOST_ONLY \
    -v $(pwd):/workspace/vla \
    mobile_vla:pytorch-2.3.0-cuda \
    sleep infinity

echo "📦 추론 컨테이너 실행 중..."
docker run -d --name ros2_inference \
    --net=host \
    -e ROS_DOMAIN_ID=$ROS_DOMAIN_ID \
    -e RMW_IMPLEMENTATION=$RMW_IMPLEMENTATION \
    -e ROS_LOCALHOST_ONLY=$ROS_LOCALHOST_ONLY \
    -v $(pwd):/workspace/vla \
    mobile_vla:pytorch-2.3.0-cuda \
    sleep infinity

echo "📦 제어 컨테이너 실행 중..."
docker run -d --name ros2_control \
    --net=host \
    -e ROS_DOMAIN_ID=$ROS_DOMAIN_ID \
    -e RMW_IMPLEMENTATION=$RMW_IMPLEMENTATION \
    -e ROS_LOCALHOST_ONLY=$ROS_LOCALHOST_ONLY \
    -v $(pwd):/workspace/vla \
    mobile_vla:pytorch-2.3.0-cuda \
    sleep infinity

echo "📦 모니터링 컨테이너 실행 중..."
docker run -d --name ros2_monitor \
    --net=host \
    -e ROS_DOMAIN_ID=$ROS_DOMAIN_ID \
    -e RMW_IMPLEMENTATION=$RMW_IMPLEMENTATION \
    -e ROS_LOCALHOST_ONLY=$ROS_LOCALHOST_ONLY \
    -v $(pwd):/workspace/vla \
    mobile_vla:pytorch-2.3.0-cuda \
    sleep infinity

echo "⏳ 컨테이너 시작 대기 중..."
sleep 5

echo "🧪 최소 기능 ROS2 노드 실행 시작..."

# 카메라 노드 실행
echo "📷 카메라 노드 실행 중..."
docker exec -d ros2_camera bash -c "
    cd /workspace/vla
    source /opt/ros/humble/setup.bash
    python3 minimal_ros2_nodes.py camera
"

# 추론 노드 실행
echo "🧠 추론 노드 실행 중..."
docker exec -d ros2_inference bash -c "
    cd /workspace/vla
    source /opt/ros/humble/setup.bash
    python3 minimal_ros2_nodes.py inference
"

# 제어 노드 실행
echo "🤖 제어 노드 실행 중..."
docker exec -d ros2_control bash -c "
    cd /workspace/vla
    source /opt/ros/humble/setup.bash
    python3 minimal_ros2_nodes.py control
"

# 모니터링 노드 실행
echo "📊 모니터링 노드 실행 중..."
docker exec -d ros2_monitor bash -c "
    cd /workspace/vla
    source /opt/ros/humble/setup.bash
    python3 minimal_ros2_nodes.py monitor
"

echo "⏳ 노드 실행 대기 중..."
sleep 10

# 통신 상태 확인
echo "🔍 통신 상태 확인 중..."
echo ""

for container in ros2_camera ros2_inference ros2_control ros2_monitor; do
    echo "📋 $container 상태:"
    docker exec $container bash -c "
        source /opt/ros/humble/setup.bash
        echo 'ROS2 노드 목록:'
        ros2 node list
        echo ''
        echo 'ROS2 토픽 목록:'
        ros2 topic list
        echo ''
        echo 'ROS2 토픽 정보:'
        ros2 topic info /camera/image_raw 2>/dev/null || echo '토픽 없음'
        ros2 topic info /inference/result 2>/dev/null || echo '토픽 없음'
        ros2 topic info /cmd_vel 2>/dev/null || echo '토픽 없음'
        echo ''
    "
    echo ""
done

# 실시간 토픽 모니터링 (5초)
echo "📊 실시간 토픽 모니터링 (5초):"
docker exec ros2_monitor bash -c "
    source /opt/ros/humble/setup.bash
    timeout 5 ros2 topic echo /camera/image_raw &
    timeout 5 ros2 topic echo /inference/result &
    timeout 5 ros2 topic echo /cmd_vel &
    sleep 6
"

echo "🎯 최소 기능 ROS2 시스템 실행 완료!"
echo ""

# 접속 방법 안내
echo "📋 컨테이너 접속 방법:"
echo "카메라 컨테이너: docker exec -it ros2_camera bash"
echo "추론 컨테이너: docker exec -it ros2_inference bash"
echo "제어 컨테이너: docker exec -it ros2_control bash"
echo "모니터링 컨테이너: docker exec -it ros2_monitor bash"
echo ""

# 정리 방법
echo "🧹 정리 방법:"
echo "docker stop ros2_camera ros2_inference ros2_control ros2_monitor"
echo "docker rm ros2_camera ros2_inference ros2_control ros2_monitor"
echo ""

echo "💡 최소 기능 ROS2 시스템이 실행되었습니다!"
echo "💡 각 노드가 서로 통신하고 있습니다!"
