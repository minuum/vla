# 🚀 Mobile VLA System - 완전 사용 가이드

## 📋 목차
1. [시스템 개요](#시스템-개요)
2. [설치 및 설정](#설치-및-설정)
3. [실행 방법](#실행-방법)
4. [테스트 및 시연](#테스트-및-시연)
5. [모니터링 및 제어](#모니터링-및-제어)
6. [문제 해결](#문제-해결)
7. [Docker 사용법](#docker-사용법)

## 🎯 시스템 개요

Mobile VLA는 Hugging Face의 `minium/mobile-vla` 모델을 사용한 Vision-Language-Action 시스템입니다.

### 🏗️ 시스템 아키텍처
```
┌─────────────────────────────────────────────────────────────────┐
│                    Mobile VLA System                           │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│  │   Camera    │    │  Inference  │    │  Action     │        │
│  │   Node      │───▶│   Node      │───▶│  Executor   │        │
│  │             │    │             │    │   Node      │        │
│  └─────────────┘    └─────────────┘    └─────────────┘        │
│         │                   │                   │              │
│         │                   │                   │              │
│         ▼                   ▼                   ▼              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│  │  System     │    │  Hugging    │    │  Robot      │        │
│  │  Monitor    │    │  Face       │    │  Control    │        │
│  │             │    │  Model      │    │             │        │
│  └─────────────┘    └─────────────┘    └─────────────┘        │
└─────────────────────────────────────────────────────────────────┘
```

### 🎯 주요 기능
- **실시간 VLM 추론**: 단일 이미지 → 18프레임 액션 예측
- **순차적 실행**: 10Hz로 액션 시퀀스 실행
- **실시간 모니터링**: 시스템 상태 및 성능 추적
- **동적 제어**: 실행 중지/재개/속도 조절

## 🔧 설치 및 설정

### 1. 환경 요구사항
```bash
# 시스템 요구사항
- Ubuntu 20.04+
- ROS2 Humble
- Python 3.8+
- NVIDIA GPU (선택사항, CUDA 11.8+)
- 8GB+ RAM
- 5GB+ 저장공간
```

### 2. ROS2 Humble 설치
```bash
# ROS2 Humble 설치
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install curl
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update
sudo apt install ros-humble-ros-base ros-humble-cv-bridge python3-colcon-common-extensions
```

### 3. Python 패키지 설치
```bash
# 필수 Python 패키지 설치
pip3 install torch torchvision torchaudio
pip3 install transformers pillow numpy opencv-python
pip3 install rclpy sensor_msgs geometry_msgs std_msgs
```

## 🚀 실행 방법

### 1. 환경 설정
```bash
# ROS 환경 설정
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=0

# Mobile VLA 패키지 환경 설정
cd /home/soda/vla/ROS_action
export AMENT_PREFIX_PATH=/home/soda/vla/ROS_action/install:$AMENT_PREFIX_PATH
```

### 2. 패키지 빌드
```bash
# Mobile VLA 패키지 빌드
colcon build --packages-select mobile_vla_package
```

### 3. 시스템 실행

#### 방법 1: 전체 시스템 실행
```bash
# 전체 시스템 실행 (카메라 시뮬레이터 포함)
ros2 launch mobile_vla_package test_mobile_vla.launch.py
```

#### 방법 2: 개별 노드 실행
```bash
# 카메라 시뮬레이터 실행
ros2 run mobile_vla_package test_camera_simulator

# 추론 노드 실행
ros2 run mobile_vla_package simple_inference_test

# 테스트 모니터 실행
ros2 run mobile_vla_package test_monitor
```

#### 방법 3: 직접 Python 실행
```bash
# 직접 실행 (환경 설정 후)
python3 src/mobile_vla_package/mobile_vla_package/test_camera_simulator.py &
python3 src/mobile_vla_package/mobile_vla_package/simple_inference_test.py &
python3 src/mobile_vla_package/mobile_vla_package/test_monitor.py &
```

## 🧪 테스트 및 시연

### 1. 시스템 상태 확인
```bash
# 노드 목록 확인
ros2 node list

# 토픽 목록 확인
ros2 topic list

# 노드 정보 확인
ros2 node info /test_camera_simulator
ros2 node info /simple_inference_test
ros2 node info /test_monitor
```

### 2. 실시간 모니터링
```bash
# 시스템 상태 모니터링
ros2 topic echo /mobile_vla/system_status

# 성능 메트릭 모니터링
ros2 topic echo /mobile_vla/performance_metrics

# 추론 결과 모니터링
ros2 topic echo /mobile_vla/inference_result
```

### 3. 카메라 이미지 확인
```bash
# 압축된 이미지 확인
ros2 topic echo /camera/image/compressed --once

# 이미지 정보 확인
ros2 topic info /camera/image/compressed
```

## 📊 모니터링 및 제어

### 1. 실시간 성능 지표
```bash
# 추론 시간 모니터링
ros2 topic echo /mobile_vla/inference_result | grep inference_time

# 실행 진행률 모니터링
ros2 topic echo /mobile_vla/execution_status | grep progress
```

### 2. 시스템 제어
```bash
# 카메라 시뮬레이터 제어
ros2 topic pub /camera_simulator/control std_msgs/msg/String "data: 'stop'"
ros2 topic pub /camera_simulator/control std_msgs/msg/String "data: 'start'"
ros2 topic pub /camera_simulator/control std_msgs/msg/String "data: 'rate:5.0'"

# 액션 실행 제어
ros2 topic pub /mobile_vla/control std_msgs/msg/String '{"action": "stop"}'
ros2 topic pub /mobile_vla/control std_msgs/msg/String '{"action": "pause"}'
ros2 topic pub /mobile_vla/control std_msgs/msg/String '{"action": "resume"}'
```

### 3. 태스크 설정
```bash
# 태스크 업데이트
ros2 topic pub /mobile_vla/task std_msgs/msg/String "data: 'Navigate around obstacles to track the target cup'"
```

## 🔧 문제 해결

### 1. 패키지를 찾을 수 없는 경우
```bash
# 환경 변수 확인
echo $AMENT_PREFIX_PATH
echo $ROS_DOMAIN_ID

# 패키지 재빌드
colcon build --packages-select mobile_vla_package --force-cmake-configure

# 환경 재설정
source install/local_setup.bash
```

### 2. 노드가 실행되지 않는 경우
```bash
# ROS 환경 확인
ros2 doctor

# 직접 실행으로 테스트
python3 src/mobile_vla_package/mobile_vla_package/test_camera_simulator.py
```

### 3. 토픽이 발행되지 않는 경우
```bash
# 노드 상태 확인
ros2 node list
ros2 node info /test_camera_simulator

# 토픽 정보 확인
ros2 topic list
ros2 topic info /camera/image/compressed
```

## 🐳 Docker 사용법

### 1. Docker 컨테이너 빌드
```bash
# Docker Compose로 빌드 및 실행
docker-compose -f docker-compose.mobile-vla.yml up --build
```

### 2. Docker 컨테이너 내부 접속
```bash
# 실행 중인 컨테이너 접속
docker exec -it mobile-vla-container bash

# 컨테이너 내부에서 시스템 실행
source /opt/ros/humble/setup.bash
cd /workspace/vla/ROS_action
colcon build --packages-select mobile_vla_package
source install/setup.bash
ros2 launch mobile_vla_package test_mobile_vla.launch.py
```

### 3. Docker 환경에서 모니터링
```bash
# 호스트에서 컨테이너 내부 토픽 모니터링
docker exec -it mobile-vla-container bash -c "source /opt/ros/humble/setup.bash && ros2 topic echo /mobile_vla/system_status"
```

## 📈 성능 최적화

### 1. GPU 가속 활성화
```bash
# CUDA 확인
nvidia-smi

# PyTorch CUDA 확인
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 2. 메모리 최적화
```bash
# 시스템 메모리 모니터링
htop

# GPU 메모리 모니터링
watch -n 1 nvidia-smi
```

### 3. 네트워크 최적화
```bash
# ROS 네트워크 설정
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0
```

## 🎯 실제 로봇 연동

### 1. 실제 카메라 연동
```bash
# USB 카메라 사용
sudo apt install v4l-utils
v4l2-ctl --list-devices

# 카메라 노드 실행
ros2 run camera_pub camera_publisher
```

### 2. 로봇 제어 연동
```bash
# 로봇 제어 노드 실행
ros2 run omni_controller omni_controller

# 액션 토픽 확인
ros2 topic echo /cmd_vel
```

### 3. 실제 환경 테스트
```bash
# 실제 환경에서 시스템 실행
ros2 launch mobile_vla_package launch_mobile_vla.launch.py inference_node:=true
```

## 📝 참고 자료

- [Mobile VLA Model (Hugging Face)](https://huggingface.co/minium/mobile-vla)
- [ROS2 Humble Documentation](https://docs.ros.org/en/humble/)
- [Transformers Library](https://huggingface.co/docs/transformers/)
- [PyTorch Documentation](https://pytorch.org/docs/)

## 🤝 기여 및 지원

### 버그 리포트
- GitHub Issues를 통해 버그를 리포트해주세요
- 상세한 로그와 재현 방법을 포함해주세요

### 기능 요청
- 새로운 기능이나 개선사항을 제안해주세요
- 구체적인 사용 사례를 설명해주세요

### 기여 방법
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

---

**🎉 Mobile VLA System을 사용해주셔서 감사합니다!**
