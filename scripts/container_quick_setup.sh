#!/bin/bash

# 🚀 도커 컨테이너 내부 ROS 환경 빠른 설정 및 빌드
# 컨테이너 환경에 최적화된 자동 설정 스크립트

set -e

echo "🚀 도커 컨테이너 내부 ROS 환경 빠른 설정 및 빌드"
echo "================================================"
echo ""

# 1. 컨테이너 환경 확인
echo "📋 1. 컨테이너 환경 확인 중..."
echo "🔍 현재 디렉토리: $(pwd)"
echo "🔍 컨테이너 ID: $(hostname)"
echo "🔍 CUDA 상태:"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
echo "🔍 GPU 정보:"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits
echo ""

# 2. ROS 환경 설정 (컨테이너용)
echo "📋 2. ROS 환경 설정 중..."
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_LOCALHOST_ONLY=1
echo "✅ ROS 환경 설정 완료"
echo ""

# 3. 워크스페이스로 이동
echo "📋 3. 워크스페이스로 이동 중..."
cd /workspace/vla/ROS_action
echo "✅ 워크스페이스 이동 완료: $(pwd)"
echo ""

# 4. 의존성 설치
echo "📋 4. 의존성 설치 중..."
echo "⚠️ rosdep 설치 중... (시간이 걸릴 수 있습니다)"
rosdep install --from-paths src --ignore-src -r -y || {
    echo "⚠️ rosdep 설치 실패, 계속 진행합니다..."
}
echo "✅ 의존성 설치 완료"
echo ""

# 5. 핵심 패키지 빌드
echo "📋 5. 핵심 패키지 빌드 중..."
echo "🔨 빌드 중인 패키지:"
echo "   - camera_interfaces"
echo "   - camera_pub"
echo "   - mobile_vla_package"
echo "   - robot_control"
echo "   - vla_inference"
echo ""

colcon build --packages-select camera_interfaces camera_pub mobile_vla_package robot_control vla_inference

if [ $? -eq 0 ]; then
    echo "✅ 핵심 패키지 빌드 완료!"
else
    echo "❌ 빌드 실패"
    exit 1
fi
echo ""

# 6. 환경 설정
echo "📋 6. 환경 설정 중..."
source install/setup.bash
echo "✅ ROS 워크스페이스 환경 설정 완료"
echo ""

# 7. 시스템 상태 확인
echo "📋 7. 시스템 상태 확인 중..."
echo "🔍 사용 가능한 패키지:"
ros2 pkg list | grep -E "(camera|mobile_vla|robot|vla)" || echo "⚠️ 패키지 목록 확인 실패"
echo ""

echo "🔍 ROS2 상태:"
if command -v ros2 &> /dev/null; then
    echo "✅ ROS2 명령어 사용 가능"
    echo "📋 ROS2 버전: $(ros2 --version)"
else
    echo "❌ ROS2 명령어를 찾을 수 없습니다"
fi
echo ""

# 8. 컨테이너 권한 확인
echo "📋 8. 컨테이너 권한 확인 중..."
echo "🔍 카메라 장치:"
ls -la /dev/video* 2>/dev/null || echo "⚠️ 카메라 장치 없음"
echo "🔍 USB 장치:"
ls -la /dev/bus/usb/ 2>/dev/null || echo "⚠️ USB 장치 없음"
echo ""

# 9. 핵심 명령어 안내
echo "📋 9. 핵심 명령어 안내:"
echo "🎯 다음 명령어들을 별도 터미널에서 실행하세요:"
echo ""
echo "📷 카메라 서비스:"
echo "  ros2 run camera_pub usb_camera_service_server"
echo ""
echo "🧠 VLA 추론 (SOTA):"
echo "  ros2 run mobile_vla_package robovlms_inference"
echo ""
echo "🤖 로봇 제어:"
echo "  ros2 run robot_control robot_control_node"
echo ""
echo "🚀 전체 시스템:"
echo "  ros2 launch mobile_vla_package robovlms_system.launch.py"
echo ""

# 10. 테스트 명령어
echo "📋 10. 테스트 명령어:"
echo "🧪 시스템 상태 확인:"
echo "  ros2 node list"
echo "  ros2 topic list"
echo "  ros2 service list"
echo ""
echo "🧪 카메라 테스트:"
echo "  ros2 service call /get_image_service camera_interfaces/srv/GetImage"
echo ""
echo "🧪 추론 테스트:"
echo "  ros2 topic echo /mobile_vla/inference_result"
echo ""

echo "🎉 도커 컨테이너 내부 ROS 환경 설정 및 빌드 완료!"
echo "💡 이제 핵심 노드들을 실행할 수 있습니다."
echo "💡 컨테이너 내부에서는 ROS_LOCALHOST_ONLY=1 설정이 적용되었습니다."
echo ""
