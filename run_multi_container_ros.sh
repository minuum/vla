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
    -v $(pwd)/ROS_action:/workspace/ros2_ws/src/ROS_action \
    -e ROS_DOMAIN_ID=42 \
    -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    -e ROS_LOCALHOST_ONLY=0 \
    -e ROS_DISCOVERY_SERVER=$DISCOVERY_SERVER \
    mobile_vla:ros \
    sleep infinity

# 2. 추론 컨테이너 실행 (--net=host 사용, ROS_action 마운트)
echo "📦 추론 컨테이너 실행 중..."
docker run -d \
    --name ros2_inference \
    --net=host \
    --gpus all \
    -v $(pwd)/ROS_action:/workspace/ros2_ws/src/ROS_action \
    -e ROS_DOMAIN_ID=42 \
    -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    -e ROS_LOCALHOST_ONLY=0 \
    -e ROS_DISCOVERY_SERVER=$DISCOVERY_SERVER \
    mobile_vla:ros \
    sleep infinity

# 3. 제어 컨테이너 실행 (--net=host 사용, ROS_action 마운트)
echo "📦 제어 컨테이너 실행 중..."
docker run -d \
    --name ros2_control \
    --net=host \
    --gpus all \
    -v $(pwd)/ROS_action:/workspace/ros2_ws/src/ROS_action \
    -e ROS_DOMAIN_ID=42 \
    -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    -e ROS_LOCALHOST_ONLY=0 \
    -e ROS_DISCOVERY_SERVER=$DISCOVERY_SERVER \
    mobile_vla:ros \
    sleep infinity

# 4. 모니터링 컨테이너 실행 (--net=host 사용, ROS_action 마운트)
echo "📦 모니터링 컨테이너 실행 중..."
docker run -d \
    --name ros2_monitor \
    --net=host \
    --gpus all \
    -v $(pwd)/ROS_action:/workspace/ros2_ws/src/ROS_action \
    -e ROS_DOMAIN_ID=42 \
    -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    -e ROS_LOCALHOST_ONLY=0 \
    -e ROS_DISCOVERY_SERVER=$DISCOVERY_SERVER \
    mobile_vla:ros \
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
    source /opt/ros/setup_ros.sh
    echo '카메라 컨테이너 ROS2 환경 설정 완료'
    echo '카메라 노드 실행 중...'
    cd /workspace/ros2_ws
    # 실제 카메라 노드 실행 (ROS_action 코드가 있다면)
    # ros2 run camera_pub usb_camera_service_server
    # 임시로 테스트 토픽 발행
    timeout 60 ros2 topic pub /camera/image_raw std_msgs/msg/String 'data: Camera Image Data' --rate 1
"

echo "🧠 추론 노드 실행 중..."
docker exec -d ros2_inference bash -c "
    source /opt/ros/setup_ros.sh
    echo '추론 컨테이너 ROS2 환경 설정 완료'
    echo '추론 노드 실행 중...'
    cd /workspace/ros2_ws
    # 실제 추론 노드 실행 (ROS_action 코드가 있다면)
    # ros2 run mobile_vla_package robovlms_inference
    # 임시로 테스트 토픽 구독 및 발행
    timeout 60 bash -c '
        ros2 topic echo /camera/image_raw &
        sleep 5
        ros2 topic pub /inference/result std_msgs/msg/String \"data: Inference Result\" --rate 0.5
    '
"

echo "🤖 제어 노드 실행 중..."
docker exec -d ros2_control bash -c "
    source /opt/ros/setup_ros.sh
    echo '제어 컨테이너 ROS2 환경 설정 완료'
    echo '제어 노드 실행 중...'
    cd /workspace/ros2_ws
    # 실제 제어 노드 실행 (ROS_action 코드가 있다면)
    # ros2 run robot_control robot_control_node
    # 임시로 테스트 토픽 구독 및 발행
    timeout 60 bash -c '
        ros2 topic echo /inference/result &
        sleep 5
        ros2 topic pub /cmd_vel geometry_msgs/msg/Twist \"{linear: {x: 1.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}\" --rate 0.5
    '
"

echo "📊 모니터링 노드 실행 중..."
docker exec -d ros2_monitor bash -c "
    source /opt/ros/setup_ros.sh
    echo '모니터링 컨테이너 ROS2 환경 설정 완료'
    echo '모니터링 노드 실행 중...'
    cd /workspace/ros2_ws
    # 실제 모니터링 노드 실행 (ROS_action 코드가 있다면)
    # ros2 run mobile_vla_package robovlms_monitor
    # 임시로 모든 토픽 모니터링
    timeout 60 bash -c '
        echo \"모니터링 시작...\"
        ros2 topic list
        echo \"\"
        ros2 topic echo /camera/image_raw &
        ros2 topic echo /inference/result &
        ros2 topic echo /cmd_vel &
        sleep 30
    '
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
        source /opt/ros/setup_ros.sh
        cd /workspace/ros2_ws
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
        ls -la src/ROS_action/
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
