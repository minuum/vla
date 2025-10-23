#!/bin/bash

echo "🚀 Mobile VLA Data Collector 시작..."
echo "=================================="

# ROS 환경 설정
cd /home/soda/vla/ROS_action
source /opt/ros/humble/setup.bash
source install/setup.bash

echo "📦 환경 설정 완료"
echo "🎯 데이터 수집기 실행 중..."

# 데이터 수집기 실행
python3 src/mobile_vla_package/mobile_vla_package/mobile_vla_data_collector.py
