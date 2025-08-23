#!/bin/bash

echo "🚀 Mobile VLA System Starting..."
echo "=================================="

# ROS 환경 설정
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=0

# 작업 디렉토리로 이동
cd /workspace/vla/ROS_action

echo "📦 Building Mobile VLA Package..."
colcon build --packages-select mobile_vla_package

if [ $? -ne 0 ]; then
    echo "❌ Build failed!"
    exit 1
fi

echo "✅ Build completed successfully!"

# 환경 설정
source install/setup.bash

echo "🔧 Environment setup completed"
echo "📊 Available topics:"
echo "  - /camera/image/compressed (input)"
echo "  - /cmd_vel (output)"
echo "  - /mobile_vla/inference_result"
echo "  - /mobile_vla/status"
echo "  - /mobile_vla/system_status"
echo "  - /mobile_vla/performance_metrics"

echo ""
echo "🎯 Starting Mobile VLA System..."
echo "Press Ctrl+C to stop"

# 시스템 실행
ros2 launch mobile_vla_package launch_mobile_vla.launch.py inference_node:=true
