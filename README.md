# 🚀 Mobile VLA + ROS2 System

PyTorch 2.3.0과 ROS2 Humble을 통합한 Vision-Language-Action (VLA) 시스템

## 📋 목차

- [시스템 개요](#시스템-개요)
- [주요 기능](#주요-기능)
- [시스템 아키텍처](#시스템-아키텍처)
- [설치 및 실행](#설치-및-실행)
- [사용법](#사용법)
- [프로젝트 구조](#프로젝트-구조)
- [문제 해결](#문제-해결)
- [업데이트 내역](#업데이트-내역)

## 🎯 시스템 개요

이 프로젝트는 Jetson 플랫폼에서 PyTorch 2.3.0과 ROS2 Humble을 통합하여 Vision-Language-Action (VLA) 시스템을 구현합니다. Kosmos-2 모델을 사용하여 이미지 기반 로봇 제어를 수행합니다.

### 🔧 기술 스택

- **PyTorch**: 2.3.0 (CUDA 지원)
- **ROS2**: Humble
- **Python**: 3.10
- **Docker**: 컨테이너화된 환경
- **CUDA**: GPU 가속
- **VLA 모델**: Kosmos-2

## ✨ 주요 기능

### 🤖 ROS2 노드 시스템
- **카메라 노드**: 실시간 이미지 캡처 및 발행
- **VLA 추론 노드**: Kosmos-2 모델을 사용한 이미지 분석 및 액션 생성
- **로봇 제어 노드**: VLA 추론 결과에 따른 로봇 제어
- **데이터 수집 노드**: 훈련 데이터 수집

### 🎮 제어 모드
- **Manual Mode**: 키보드 수동 제어
- **VLA Mode**: AI 기반 자동 제어
- **Hybrid Mode**: 수동 + AI 혼합 제어

### 📊 데이터 관리
- HDF5 형식 데이터 저장
- 이미지 및 액션 데이터 수집
- 에피소드 기반 데이터 구조

## 🏗️ 시스템 아키텍처

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Camera Node   │───▶│  VLA Inference  │───▶│ Robot Control   │
│                 │    │     Node        │    │     Node        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Image Topic    │    │ Action Topic    │    │  Cmd Vel Topic  │
│ /camera/image   │    │ /vla_action     │    │    /cmd_vel     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 설치 및 실행

### 1. 사전 요구사항

- NVIDIA Jetson (ARM64 아키텍처)
- Docker
- CUDA 지원
- ROS2 Humble

### 2. 프로젝트 클론

```bash
git clone <repository-url>
cd vla
```

### 3. Docker 이미지 빌드

```bash
# PyTorch 2.3.0 + ROS2 Humble 이미지 빌드
docker build -t mobile_vla:pytorch-2.3.0-ros2-complete -f Dockerfile.mobile-vla .
```

### 4. 컨테이너 실행

```bash
# GPU 지원으로 컨테이너 실행
docker run --rm --gpus all \
  -v /usr/local/cuda:/usr/local/cuda:ro \
  -v /usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu:ro \
  -v $(pwd)/ROS_action:/workspace/vla/ROS_action \
  -v $(pwd)/mobile_vla_dataset:/workspace/vla/mobile_vla_dataset \
  --network host --privileged -it \
  mobile_vla:pytorch-2.3.0-ros2-complete bash
```

## 📖 사용법

### 컨테이너 내부 실행

```bash
# 1. ROS 환경 설정
source /opt/ros/humble/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DISTRO=humble
export LD_LIBRARY_PATH=/opt/ros/humble/lib:$LD_LIBRARY_PATH

# 2. ROS 워크스페이스 빌드
cd /workspace/vla/ROS_action
colcon build

# 3. 환경 설정
source install/setup.bash

# 4. Mobile VLA 시스템 실행
ros2 launch launch_mobile_vla.launch.py
```

### 실행 옵션

```bash
# 카메라 노드만 실행
ros2 launch launch_mobile_vla.launch.py camera_node:=true inference_node:=false control_node:=false

# 추론 노드만 실행
ros2 launch launch_mobile_vla.launch.py camera_node:=false inference_node:=true control_node:=false

# 전체 시스템 실행
ros2 launch launch_mobile_vla.launch.py camera_node:=true inference_node:=true control_node:=true

# 데이터 수집 모드
ros2 launch launch_mobile_vla.launch.py data_collector:=true
```

### 스크립트 사용

```bash
# 컨테이너 내부 메뉴 시스템
container-run

# 또는 직접 실행
run-mobile-vla
```

## 📁 프로젝트 구조

```
vla/
├── Dockerfile.mobile-vla              # Docker 이미지 설정
├── docker-compose.yml                 # Docker Compose 설정
├── scripts/
│   ├── run_mobile_vla.sh             # 메인 실행 스크립트
│   └── container_run.sh              # 컨테이너 내부 실행 스크립트
├── ROS_action/                       # ROS2 워크스페이스
│   ├── launch/
│   │   └── launch_mobile_vla.launch.py  # 메인 launch 파일
│   └── src/
│       ├── mobile_vla_package/       # 메인 VLA 패키지
│       │   ├── vla_inference_node.py     # VLA 추론 노드
│       │   ├── robot_control_node.py     # 로봇 제어 노드
│       │   ├── mobile_vla_data_collector.py  # 데이터 수집 노드
│       │   └── ros_env_setup.py          # ROS 환경 설정 노드
│       └── camera_pub/               # 카메라 패키지
│           └── camera_publisher_continuous.py
├── mobile_vla_dataset/               # 데이터셋 저장소
└── README.md                         # 이 파일
```

## 🔧 문제 해결

### RMW 구현체 오류

```bash
# 오류: rmw_cyclonedx_cpp not found
# 해결: FastRTPS 사용
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
```

### 라이브러리 경로 문제

```bash
# 오류: librmw_cyclonedx_cpp.so not found
# 해결: LD_LIBRARY_PATH 설정
export LD_LIBRARY_PATH=/opt/ros/humble/lib:$LD_LIBRARY_PATH
```

### ROS 환경 설정

```bash
# 모든 ROS 환경 변수 설정
source /opt/ros/humble/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DISTRO=humble
export LD_LIBRARY_PATH=/opt/ros/humble/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/opt/ros/humble/lib/python3.10/site-packages:$PYTHONPATH
export ROS_DOMAIN_ID=42
export ROS_LOCALHOST_ONLY=0
```

### Docker 빌드 문제

```bash
# 캐시 삭제 후 재빌드
docker system prune -a
docker build --no-cache -t mobile_vla:pytorch-2.3.0-ros2-complete -f Dockerfile.mobile-vla .
```

## 📈 업데이트 내역

### v1.0.0 (2024-08-21)
- ✅ PyTorch 2.3.0 + ROS2 Humble 통합
- ✅ Docker 컨테이너화 완료
- ✅ RMW 구현체 문제 해결 (FastRTPS 사용)
- ✅ ROS 환경 설정 자동화
- ✅ Launch 파일 시스템 구축
- ✅ 데이터 수집 시스템 구현
- ✅ 로봇 제어 시스템 구현

### 주요 해결 사항
1. **Python 3.8 → 3.10 업그레이드**: PyTorch 2.3.0 호환성
2. **RMW 구현체 변경**: cyclonedx → fastrtps
3. **라이브러리 경로 설정**: LD_LIBRARY_PATH, PYTHONPATH
4. **ROS 환경 변수 설정**: 자동화된 환경 설정
5. **Docker 최적화**: 불필요한 패키지 제거로 이미지 크기 감소

## 🤝 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 📞 연락처

프로젝트 링크: [https://github.com/your-username/mobile-vla-ros2](https://github.com/your-username/mobile-vla-ros2)

---

**🚀 Mobile VLA + ROS2 System이 성공적으로 구축되었습니다!** 
