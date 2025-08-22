# 🚀 Mobile VLA System

Hugging Face의 `minium/mobile-vla` 모델을 사용한 Vision-Language-Action 시스템입니다.

## 🎯 주요 기능

- **실시간 VLM 추론**: 단일 이미지 → 18프레임 액션 예측
- **순차적 실행**: 10Hz로 액션 시퀀스 실행
- **실시간 모니터링**: 시스템 상태 및 성능 추적
- **동적 제어**: 실행 중지/재개/속도 조절
- **Docker 지원**: 완전한 컨테이너화된 실행 환경

## 🏗️ 시스템 아키텍처

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

## 🚀 빠른 시작

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
colcon build --packages-select mobile_vla_package
```

### 3. 시스템 실행
```bash
# 전체 시스템 실행 (카메라 시뮬레이터 포함)
ros2 launch mobile_vla_package test_mobile_vla.launch.py
```

## 📊 실시간 모니터링

```bash
# 시스템 상태 확인
ros2 topic echo /mobile_vla/system_status

# 성능 메트릭 확인
ros2 topic echo /mobile_vla/performance_metrics

# 추론 결과 확인
ros2 topic echo /mobile_vla/inference_result
```

## 🐳 Docker 실행

```bash
# Docker 컨테이너 빌드 및 실행
docker-compose -f docker-compose.mobile-vla.yml up --build
```

## 📈 성능 지표

- **모델**: minium/mobile-vla (Hugging Face)
- **입력**: RGB 이미지 (224x224) + 텍스트 태스크
- **출력**: 18프레임 액션 시퀀스 [linear_x, linear_y, angular_z]
- **추론 시간**: 100-500ms (GPU 기준)
- **실행 시간**: 1.8초 (18프레임 × 100ms)
- **프레임 레이트**: 10Hz

## 📁 프로젝트 구조

```
ROS_action/
├── src/
│   └── mobile_vla_package/
│       ├── mobile_vla_package/
│       │   ├── mobile_vla_inference.py      # VLM 추론 노드
│       │   ├── action_sequence_executor.py  # 액션 실행 노드
│       │   ├── system_monitor.py           # 시스템 모니터링
│       │   ├── test_camera_simulator.py    # 테스트 카메라
│       │   ├── test_monitor.py             # 테스트 모니터
│       │   └── simple_inference_test.py    # 간단한 추론 테스트
│       ├── launch/
│       │   ├── launch_mobile_vla.launch.py # 메인 launch 파일
│       │   └── test_mobile_vla.launch.py   # 테스트 launch 파일
│       └── requirements.txt                # Python 의존성
├── docker-compose.mobile-vla.yml           # Docker Compose
├── Dockerfile.mobile-vla                   # Docker 이미지
├── start_mobile_vla.sh                     # 시작 스크립트
└── MOBILE_VLA_USAGE_GUIDE.md              # 상세 사용 가이드
```

## 🎯 사용 예시

### 기본 실행
```bash
# 1. 환경 설정
source /opt/ros/humble/setup.bash
export AMENT_PREFIX_PATH=/home/soda/vla/ROS_action/install:$AMENT_PREFIX_PATH
export ROS_DOMAIN_ID=0

# 2. 시스템 실행
ros2 launch mobile_vla_package test_mobile_vla.launch.py
```

### 개별 노드 실행
```bash
# 카메라 시뮬레이터
ros2 run mobile_vla_package test_camera_simulator

# 추론 노드
ros2 run mobile_vla_package simple_inference_test

# 테스트 모니터
ros2 run mobile_vla_package test_monitor
```

### 시스템 제어
```bash
# 카메라 시뮬레이터 제어
ros2 topic pub /camera_simulator/control std_msgs/msg/String "data: 'stop'"

# 액션 실행 제어
ros2 topic pub /mobile_vla/control std_msgs/msg/String '{"action": "stop"}'

# 태스크 설정
ros2 topic pub /mobile_vla/task std_msgs/msg/String "data: 'Navigate around obstacles'"
```

## 🔧 시스템 요구사항

- **OS**: Ubuntu 20.04+
- **ROS**: ROS2 Humble
- **Python**: 3.8+
- **GPU**: NVIDIA GPU (선택사항, CUDA 11.8+)
- **메모리**: 8GB+ RAM
- **저장공간**: 5GB+

## 📝 참고 자료

- [Mobile VLA Model (Hugging Face)](https://huggingface.co/minium/mobile-vla)
- [상세 사용 가이드](MOBILE_VLA_USAGE_GUIDE.md)
- [시스템 시퀀스 다이어그램](Mobile_VLA_System_Sequence.md)

## 🤝 기여

버그 리포트나 기능 요청은 GitHub Issues를 통해 제출해주세요.

## 📄 라이선스

이 프로젝트는 Apache 2.0 라이선스 하에 배포됩니다.

---

**🎉 Mobile VLA System을 사용해주셔서 감사합니다!**
