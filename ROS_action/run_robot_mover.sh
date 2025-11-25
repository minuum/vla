#!/bin/bash

echo "🤖 Simple Robot Mover 시작..."
echo "============================="

# ROS 환경 설정
cd /home/soda/vla/ROS_action
source /opt/ros/humble/setup.bash
source install/setup.bash

echo "📦 환경 설정 완료"
echo "🎮 로봇 제어기 실행 중..."
echo "📋 조작법: WASD (이동), QEZC (대각선), RT (회전), 스페이스바 (정지)"

# 로봇 제어기 실행
python3 src/mobile_vla_package/mobile_vla_package/simple_robot_mover.py
