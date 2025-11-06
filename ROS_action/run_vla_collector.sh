#!/bin/bash

echo "🚀 Mobile VLA Data Collector 시작..."
echo "=================================="

# ROS 환경 설정 (절대 경로 사용, 현재 작업 디렉토리 무관)
ROS_ACTION_DIR="/home/soda/vla/ROS_action"

# ROS_action 디렉토리로 이동 (존재 확인)
if [ ! -d "$ROS_ACTION_DIR" ]; then
    echo "❌ ROS_action 디렉토리를 찾을 수 없습니다: $ROS_ACTION_DIR"
    exit 1
fi

cd "$ROS_ACTION_DIR" || exit 1

# ROS2 기본 환경 소싱
if [ -f "/opt/ros/humble/setup.bash" ]; then
    source /opt/ros/humble/setup.bash
else
    echo "❌ ROS2 Humble이 설치되지 않았습니다."
    exit 1
fi

# install 폴더가 있으면 install 폴더로 이동 후 local_setup.bash 소싱
INSTALL_DIR="$ROS_ACTION_DIR/install"
if [ -d "$INSTALL_DIR" ]; then
    if [ -f "$INSTALL_DIR/local_setup.bash" ]; then
        # install 폴더로 이동 후 source local_setup.bash
        cd "$INSTALL_DIR" || exit 1
        source local_setup.bash
        cd "$ROS_ACTION_DIR" || exit 1
        echo "✅ ROS2 워크스페이스 환경 설정 완료"
    elif [ -f "$INSTALL_DIR/setup.bash" ]; then
        # setup.bash가 있으면 그것도 사용
        cd "$INSTALL_DIR" || exit 1
        source setup.bash
        cd "$ROS_ACTION_DIR" || exit 1
        echo "✅ ROS2 워크스페이스 환경 설정 완료"
    else
        echo "⚠️  install 폴더는 있지만 setup 파일이 없습니다."
    fi
else
    echo "⚠️  ROS2 워크스페이스가 빌드되지 않았습니다 (install 폴더 없음)"
    echo "💡 절대 경로로 직접 실행합니다..."
fi

echo "📦 환경 설정 완료"
echo "🎯 데이터 수집기 실행 중..."

# 데이터 수집기 실행 (절대 경로 사용)
python3 "$ROS_ACTION_DIR/src/mobile_vla_package/mobile_vla_package/mobile_vla_data_collector.py"
