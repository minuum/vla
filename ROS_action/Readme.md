# ROS_action 패키지

**Vision-Language-Action (VLA) 기반 로봇 제어 시스템**

이 프로젝트는 ROS2 기반의 멀티모달 로봇 제어 시스템으로, 시각, 음성, 액션 처리를 통합한 VLA 파이프라인을 구현합니다.

## 📁 패키지 구조

```
ROS_action/
├── src/
│   ├── camera_pub/         # 카메라 이미지 퍼블리셔
│   ├── camera_sub/         # 카메라 이미지 구독자
│   ├── mic_pub/           # 마이크 오디오 퍼블리셔
│   ├── object_pose/       # 객체 탐지 및 위치 추정
│   ├── omni_controller/   # 옴니휠 로봇 제어
│   └── ros_action_msgs/   # 커스텀 메시지 타입
├── install/
└── log/
```

## 🔧 패키지 상세 설명

### 1. camera_pub 📷
**Jetson Orin IMX219 카메라 퍼블리셔**

- **기능**: CSI 카메라를 통해 실시간 이미지 캡처 및 퍼블리시
- **토픽**: `/camera/image_raw` (sensor_msgs/Image)
- **특징**: 
  - GStreamer 파이프라인 사용
  - 1280x720 해상도, 30fps
  - 180도 회전 처리
  - Enter 키 기반 수동 캡처 모드

### 2. camera_sub 📸
**카메라 이미지 구독 및 저장**

- **기능**: 카메라 이미지를 받아 로컬 디스크에 저장
- **구독 토픽**: `/camera/image_raw` (sensor_msgs/Image)
- **저장 경로**: `~/camera_images/`
- **특징**: 타임스탬프 기반 파일명 생성

### 3. mic_pub 🎤
**마이크 오디오 퍼블리셔**

- **기능**: 시스템 마이크를 통한 오디오 녹음 및 퍼블리시
- **토픽**: `/audio/raw` (std_msgs/Int16MultiArray)
- **설정**: 16kHz 샘플링, 16-bit, 모노
- **특징**: 
  - 5초 간격 녹음
  - Enter 키 기반 수동 녹음 모드
  - WAV 파일 저장 옵션 (주석 처리됨)

### 4. object_pose 🎯
**YOLOv5 기반 객체 탐지 및 위치 추정**

- **기능**: 
  - YOLOv5s 모델을 사용한 실시간 객체 탐지
  - 바운딩박스 기반 거리 및 각도 계산
  - 음성 인식 키워드 기반 타겟 객체 전환
- **구독 토픽**: 
  - `/camera/image_raw` (sensor_msgs/Image)
  - `/stt/text` (std_msgs/String)
- **퍼블리시 토픽**: `/object_info` (ros_action_msgs/ObjectInfo)
- **지원 객체**: 사람(person), 의자(chair), 노트북(laptop)
- **특징**:
  - GPU/CPU 자동 감지
  - 카메라 시야각 110도 기준 각도 계산
  - 객체 높이 0.15m 가정한 거리 추정

### 5. omni_controller 🤖
**옴니휠 로봇 제어 노드**

- **기능**: 객체 위치 정보를 기반으로 한 로봇 이동 제어
- **구독 토픽**: `/object_info` (ros_action_msgs/ObjectInfo)
- **하드웨어**: 
  - POP 플랫폼 기반 옴니휠 로봇
  - PSD 센서 (근접 센서)
  - 초음파 센서
- **제어 로직**:
  - 각도 기반 회전 제어 (92도/초 기준)
  - 장애물 감지 시 자동 정지 (20cm 이내)
  - 회전 후 전진 이동

### 6. ros_action_msgs 📝
**커스텀 메시지 타입 정의**

#### ObjectInfo.msg
```
string object_id     # 객체 식별자
float64 distance     # 거리 (미터)
float64 angle        # 각도 (도, 정면 기준)
```

## 🚀 빌드 및 실행

### 빌드
```bash
cd ROS_action
colcon build
source install/setup.bash
```

### 실행 순서

1. **카메라 스트리밍 시작**
```bash
ros2 run camera_pub camera_publisher_node
```

2. **객체 탐지 노드 실행**
```bash
ros2 run object_pose object_pose_publisher
```

3. **로봇 제어 노드 실행**
```bash
ros2 run omni_controller omni_drive_node
```

4. **옵션: 이미지 저장 (디버깅용)**
```bash
ros2 run camera_sub camera_subscriber_node
```

5. **옵션: 오디오 녹음 (STT 연동용)**
```bash
ros2 run mic_pub mic_publisher_node
```

## 📊 데이터 플로우

```
카메라 → /camera/image_raw → 객체 탐지 → /object_info → 로봇 제어
마이크 → /audio/raw → STT → /stt/text → 객체 전환
```

## 🔧 시스템 요구사항

- **OS**: Ubuntu 22.04 (ROS2 Humble)
- **Hardware**: 
  - Jetson Orin (권장) 또는 NVIDIA GPU
  - IMX219 CSI 카메라
  - POP 플랫폼 옴니휠 로봇
  - 마이크 (USB 또는 내장)
- **Dependencies**:
  - OpenCV
  - PyTorch
  - YOLOv5
  - cv_bridge
  - POP 라이브러리

## 🔗 VLA 파이프라인 연동

이 ROS_action 패키지는 `RoboVLMs/vla_test/action_parser.py`와 연동되어 다음과 같은 VLA 파이프라인을 구성합니다:

1. **Vision**: 카메라 이미지 입력
2. **Language**: 음성 인식 텍스트 입력
3. **Action**: VLA 모델 추론 결과를 로봇 액션으로 변환

## 🐛 문제 해결

### 카메라 관련
- GStreamer 파이프라인 오류 시 `nvarguscamerasrc` 지원 여부 확인
- 권한 문제 시 사용자를 `video` 그룹에 추가

### 마이크 관련
- `arecord` 명령어 지원 여부 확인
- ALSA 설정 확인

### GPU 관련
- CUDA 설치 및 PyTorch GPU 지원 확인
- 메모리 부족 시 배치 크기 조정

## 📅 최근 업데이트

- **2025.03**: 초기 패키지 구조 설계
- **2025.04**: YOLOv5 기반 객체 탐지 구현
- **2025.05**: 옴니휠 로봇 제어 로직 추가
- **2025.05**: 음성 인식 키워드 기반 객체 전환 기능 추가
- **2025.06**: VLA 파이프라인 연동 준비

## 👥 기여자

- **카메라 시스템**: soda
- **마이크 시스템**: shosae
- **객체 탐지**: shosae
- **로봇 제어**: 팀 공동 작업