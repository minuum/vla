#!/bin/bash

# 🚀 다중 컨테이너 ROS2 시스템 실행 스크립트
# 여러 컨테이너에서 각각 다른 ROS2 노드들을 실행

set -e

echo "🚀 다중 컨테이너 ROS2 시스템 실행"
echo "=================================="
echo ""

# 기존 컨테이너 정리
echo "🧹 기존 컨테이너 정리 중..."
docker stop ros2_camera ros2_inference ros2_control ros2_monitor 2>/dev/null || true
docker rm ros2_camera ros2_inference ros2_control ros2_monitor 2>/dev/null || true

# Discovery Server 설정
DISCOVERY_SERVER=":11811"

# 1. 카메라 컨테이너 실행 (--net=host 사용, ROS_action 마운트)
echo "📦 카메라 컨테이너 실행 중..."
docker run -d \
    --name ros2_camera \
    --net=host \
    --gpus all \
    -v /dev/bus/usb:/dev/bus/usb \
    -v $(pwd):/workspace/vla \
    -e ROS_DOMAIN_ID=42 \
    -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    -e ROS_LOCALHOST_ONLY=0 \
    -e ROS_DISCOVERY_SERVER=$DISCOVERY_SERVER \
    mobile_vla:pytorch-2.3.0-cuda \
    sleep infinity

# 2. 추론 컨테이너 실행 (--net=host 사용, ROS_action 마운트)
echo "📦 추론 컨테이너 실행 중..."
docker run -d \
    --name ros2_inference \
    --net=host \
    --gpus all \
    -v $(pwd):/workspace/vla \
    -e ROS_DOMAIN_ID=42 \
    -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    -e ROS_LOCALHOST_ONLY=0 \
    -e ROS_DISCOVERY_SERVER=$DISCOVERY_SERVER \
    mobile_vla:pytorch-2.3.0-cuda \
    sleep infinity

# 3. 제어 컨테이너 실행 (--net=host 사용, ROS_action 마운트)
echo "📦 제어 컨테이너 실행 중..."
docker run -d \
    --name ros2_control \
    --net=host \
    --gpus all \
    -v $(pwd):/workspace/vla \
    -e ROS_DOMAIN_ID=42 \
    -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    -e ROS_LOCALHOST_ONLY=0 \
    -e ROS_DISCOVERY_SERVER=$DISCOVERY_SERVER \
    mobile_vla:pytorch-2.3.0-cuda \
    sleep infinity

# 4. 모니터링 컨테이너 실행 (--net=host 사용, ROS_action 마운트)
echo "📦 모니터링 컨테이너 실행 중..."
docker run -d \
    --name ros2_monitor \
    --net=host \
    --gpus all \
    -v $(pwd):/workspace/vla \
    -e ROS_DOMAIN_ID=42 \
    -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    -e ROS_LOCALHOST_ONLY=0 \
    -e ROS_DISCOVERY_SERVER=$DISCOVERY_SERVER \
    mobile_vla:pytorch-2.3.0-cuda \
    sleep infinity

# 컨테이너 상태 확인
echo "📋 컨테이너 상태 확인:"
sleep 3
docker ps | grep ros2_

echo ""
echo "🧪 ROS2 노드 실행 시작..."
echo ""

# 각 컨테이너에서 ROS2 노드 실행
echo "📷 카메라 노드 실행 중..."
docker exec -d ros2_camera bash -c "
    cd /workspace/vla/ROS_action
    source /opt/ros/humble/setup.bash
    source install/setup.bash
    python3 src/camera_pub/camera_pub/camera_publisher_continuous.py
"

echo "🧠 추론 노드 실행 중..."
docker exec -d ros2_inference bash -c "
    cd /workspace/vla/ROS_action
    source /opt/ros/humble/setup.bash
    source install/setup.bash
    python3 src/mobile_vla_package/mobile_vla_package/robovlms_inference.py
"

echo "🤖 제어 노드 실행 중..."
docker exec -d ros2_control bash -c "
    cd /workspace/vla/ROS_action
    source /opt/ros/humble/setup.bash
    source install/setup.bash
    python3 src/mobile_vla_package/mobile_vla_package/simple_robot_mover.py
"

echo "📊 모니터링 노드 실행 중..."
docker exec -d ros2_monitor bash -c "
    cd /workspace/vla/ROS_action
    source /opt/ros/humble/setup.bash
    source install/setup.bash
    python3 src/mobile_vla_package/mobile_vla_package/robovlms_monitor.py
"

# 잠시 대기
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
        echo 'ROS2 서비스 목록:'
        ros2 service list
        echo ''
        echo 'ROS_action 디렉토리 확인:'
        ls -la /workspace/vla/ROS_action/
        echo ''
    "
    echo ""
done

# 로그 확인
echo "📋 각 컨테이너 로그 확인:"
for container in ros2_camera ros2_inference ros2_control ros2_monitor; do
    echo "📋 $container 로그:"
    docker logs $container --tail 5
    echo ""
done

echo "🎯 다중 컨테이너 ROS2 시스템 실행 완료!"
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

echo "💡 성공적으로 다중 컨테이너 ROS2 시스템이 실행되었습니다!"
echo "💡 각 컨테이너에서 ROS2 노드들이 서로 통신하고 있습니다!"
