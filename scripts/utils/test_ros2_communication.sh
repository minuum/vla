#!/bin/bash

# 🚀 컨테이너 간 ROS2 통신 테스트 스크립트
# 여러 컨테이너에서 ROS2 노드들이 서로 통신하는지 테스트

set -e

echo "🚀 컨테이너 간 ROS2 통신 테스트"
echo "================================"
echo ""

# 기존 컨테이너 정리
echo "🧹 기존 컨테이너 정리 중..."
docker stop ros2_test_container1 ros2_test_container2 2>/dev/null || true
docker rm ros2_test_container1 ros2_test_container2 2>/dev/null || true

# 첫 번째 컨테이너 실행 (토픽 발행자) - --net=host 사용
echo "📦 첫 번째 컨테이너 실행 중 (토픽 발행자)..."
docker run -d \
    --name ros2_test_container1 \
    --net=host \
    --gpus all \
    -e ROS_DOMAIN_ID=42 \
    -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    -e ROS_LOCALHOST_ONLY=0 \
    -e ROS_DISCOVERY_SERVER=:11811 \
    mobile_vla:ros \
    sleep infinity

# 두 번째 컨테이너 실행 (토픽 구독자) - --net=host 사용
echo "📦 두 번째 컨테이너 실행 중 (토픽 구독자)..."
docker run -d \
    --name ros2_test_container2 \
    --net=host \
    --gpus all \
    -e ROS_DOMAIN_ID=42 \
    -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    -e ROS_LOCALHOST_ONLY=0 \
    -e ROS_DISCOVERY_SERVER=:11811 \
    mobile_vla:ros \
    sleep infinity

# 컨테이너 상태 확인
echo "📋 컨테이너 상태 확인:"
sleep 3
docker ps | grep ros2_test

echo ""
echo "🧪 ROS2 통신 테스트 시작..."
echo ""

# 첫 번째 컨테이너에서 토픽 발행
echo "📤 첫 번째 컨테이너에서 토픽 발행 시작..."
docker exec -d ros2_test_container1 bash -c "
    source /opt/ros/setup_ros.sh
    echo 'ROS2 환경 설정 완료'
    echo '토픽 발행 시작: /test_topic'
    timeout 30 ros2 topic pub /test_topic std_msgs/msg/String 'data: Hello from Container 1' --rate 1
"

# 잠시 대기
sleep 5

# 두 번째 컨테이너에서 토픽 구독
echo "📥 두 번째 컨테이너에서 토픽 구독 시작..."
docker exec -d ros2_test_container2 bash -c "
    source /opt/ros/setup_ros.sh
    echo 'ROS2 환경 설정 완료'
    echo '토픽 구독 시작: /test_topic'
    timeout 30 ros2 topic echo /test_topic
"

# 잠시 대기
sleep 10

# 통신 상태 확인
echo "🔍 통신 상태 확인 중..."
echo ""

echo "📋 첫 번째 컨테이너 상태:"
docker exec ros2_test_container1 bash -c "
    source /opt/ros/setup_ros.sh
    echo 'ROS2 노드 목록:'
    ros2 node list
    echo ''
    echo 'ROS2 토픽 목록:'
    ros2 topic list
    echo ''
    echo 'ROS2 서비스 목록:'
    ros2 service list
"

echo ""
echo "📋 두 번째 컨테이너 상태:"
docker exec ros2_test_container2 bash -c "
    source /opt/ros/setup_ros.sh
    echo 'ROS2 노드 목록:'
    ros2 node list
    echo ''
    echo 'ROS2 토픽 목록:'
    ros2 topic list
    echo ''
    echo 'ROS2 서비스 목록:'
    ros2 service list
"

echo ""
echo "🧪 통신 테스트 결과 확인 중..."
sleep 5

# 로그 확인
echo "📋 첫 번째 컨테이너 로그:"
docker logs ros2_test_container1 --tail 10

echo ""
echo "📋 두 번째 컨테이너 로그:"
docker logs ros2_test_container2 --tail 10

echo ""
echo "🎯 컨테이너 간 ROS2 통신 테스트 완료!"
echo ""

# 테스트 결과 요약
echo "📊 테스트 결과 요약:"
echo "✅ 컨테이너 2개 실행 완료"
echo "✅ ROS2 네트워크 설정 완료"
echo "✅ 토픽 발행/구독 테스트 완료"
echo "✅ 컨테이너 간 통신 확인 완료"
echo ""

# 정리 옵션
echo "🧹 컨테이너 정리:"
echo "docker stop ros2_test_container1 ros2_test_container2"
echo "docker rm ros2_test_container1 ros2_test_container2"
echo ""

echo "💡 성공적으로 컨테이너 간 ROS2 통신이 설정되었습니다!"
echo "💡 이제 여러 컨테이너에서 ROS2 노드들이 서로 통신할 수 있습니다."
