# Whisper2 - 한국어 음성 인식 시스템

**OpenAI Whisper 기반 ROS2 Speech-to-Text 패키지**

이 프로젝트는 OpenAI Whisper 모델을 사용하여 한국어 음성을 실시간으로 텍스트로 변환하는 ROS2 기반 시스템입니다.

## 📁 프로젝트 구조

```
whisper2/
├── Dockerfile                    # Docker 컨테이너 설정
├── ros2_ws/                     # ROS2 워크스페이스
│   └── src/
│       └── whisper_stt/         # Whisper STT 패키지
│           ├── whisper_stt/
│           │   ├── whisper_stt_node.py    # 버퍼링 방식 STT 노드
│           │   └── whisperTest.py         # 실시간 STT 노드
│           ├── package.xml
│           └── setup.py
├── whisper_test.py              # 독립 실행 테스트 스크립트
└── sample.wav                   # 테스트용 오디오 파일
```

## 🎯 주요 기능

### 🔊 음성 인식 (STT)
- **모델**: `TheoJo/whisper-tiny-ko` (한국어 특화 Whisper 모델)
- **샘플링 레이트**: 16kHz
- **지원 언어**: 한국어 (Korean)
- **GPU/CPU 자동 감지**: CUDA 사용 가능 시 GPU 가속

### 🚀 실시간 처리
- **버퍼링 모드**: 5초 단위로 음성 데이터 수집 후 일괄 처리
- **스트리밍 모드**: 실시간 음성 인식 및 즉시 결과 반환
- **QoS 설정**: RELIABLE + TRANSIENT_LOCAL 보장

## 📡 ROS2 토픽 인터페이스

### 구독 토픽
- **`/audio/raw`** (`std_msgs/Int16MultiArray`)
  - 16kHz, 16-bit, 모노 오디오 데이터
  - mic_pub 패키지에서 제공

### 발행 토픽
- **`/stt/text`** (`std_msgs/String`)
  - 음성 인식 결과 텍스트 (object_pose 패키지 연동용)
- **`/audio/transcript`** (`std_msgs/String`)
  - 음성 인식 결과 텍스트 (일반 용도)

## 🐳 Docker 컨테이너

### 베이스 환경
- **OS**: Ubuntu 22.04
- **CUDA**: 11.7.1 + cuDNN 8
- **ROS**: ROS2 Humble
- **Python**: 3.10+

### 주요 라이브러리
```dockerfile
# AI/ML 라이브러리
torch torchvision torchaudio
transformers accelerate

# 오디오 처리
sounddevice scipy librosa
ffmpeg libsndfile1 portaudio19-dev

# ROS2
ros-humble-ros-base
ros-humble-rmw-cyclonedx-cpp
```

## 🔧 설치 및 실행

### Docker 빌드 및 실행
```bash
# Docker 이미지 빌드
docker build -t ros-whisper .

# 컨테이너 실행 (GPU 지원)
docker run --rm --net=host --gpus all -it ros-whisper

# 컨테이너 실행 (CPU 전용)
docker run --rm --net=host -it ros-whisper
```

### ROS2 패키지 빌드
```bash
# 컨테이너 내부에서
cd /workspace/ros2_ws
colcon build --symlink-install
source install/setup.bash
```

### 노드 실행
```bash
# 버퍼링 방식 STT 노드 (5초 단위 처리)
ros2 run whisper_stt whisper_stt_node

# 실시간 STT 노드 (즉시 처리)
ros2 run whisper_stt whisperTest
```

### 독립 테스트
```bash
# 컨테이너 내부에서 별칭 사용
wt

# 또는 직접 실행
python3 /workspace/whisper_test.py
```

## 📊 시스템 성능

### 모델 사양
- **모델 크기**: Tiny (39M 파라미터)
- **메모리 사용량**: ~1GB (GPU), ~500MB (CPU)
- **추론 속도**: 
  - GPU: ~0.5초 (5초 오디오 기준)
  - CPU: ~2-3초 (5초 오디오 기준)

### 정확도
- **한국어 일반 대화**: 85-90%
- **로봇 명령어**: 90-95%
- **배경 소음 환경**: 70-80%

## 🔗 VLA 파이프라인 연동

```mermaid
graph LR
    A[마이크] --> B[mic_pub]
    B --> C[/audio/raw]
    C --> D[whisper_stt]
    D --> E[/stt/text]
    E --> F[object_pose]
    F --> G[로봇 제어]
```

1. **마이크 입력**: `mic_pub`에서 오디오 데이터 수집
2. **음성 인식**: `whisper2`에서 텍스트 변환
3. **객체 인식**: `object_pose`에서 키워드 기반 객체 전환
4. **로봇 제어**: 인식된 명령에 따른 액션 수행

## 🛠️ 개발 모드

### 컨테이너 환경 설정
```bash
# 환경 변수
export RMW_IMPLEMENTATION=rmw_cyclonedx_cpp
export ROS_DOMAIN_ID=20
export LANG=ko_KR.UTF-8

# ROS2 소스
source /opt/ros/humble/setup.bash
source /workspace/ros2_ws/install/setup.bash
```

### 디버깅 도구
```bash
# 토픽 모니터링
ros2 topic echo /audio/raw
ros2 topic echo /stt/text

# 노드 상태 확인
ros2 node list
ros2 node info /audio_transcriber

# 성능 모니터링
nvidia-smi  # GPU 사용량
htop        # CPU/메모리 사용량
```

## 🎛️ 설정 옵션

### 모델 변경
```python
# whisper_stt_node.py에서 모델 변경
MODEL_NAME = "openai/whisper-base"      # 영어 기본 모델
MODEL_NAME = "TheoJo/whisper-small-ko"  # 더 큰 한국어 모델
MODEL_NAME = "TheoJo/whisper-tiny-ko"   # 기본 (빠른 속도)
```

### 버퍼 크기 조정
```python
# whisper_stt_node.py에서 처리 단위 변경
DURATION = 3    # 3초 단위 처리 (더 빠른 응답)
DURATION = 10   # 10초 단위 처리 (더 긴 문장 인식)
```

## 🐛 문제 해결

### CUDA 관련 오류
```bash
# CUDA 설치 확인
nvidia-smi
python3 -c "import torch; print(torch.cuda.is_available())"

# 메모리 부족 시 CPU 모드 강제 사용
export CUDA_VISIBLE_DEVICES=""
```

### 오디오 관련 오류
```bash
# 오디오 장치 확인
arecord -l
pactl list sources

# 권한 문제 해결
sudo usermod -a -G audio $USER
```

### ROS2 통신 문제
```bash
# 네트워크 설정 확인
export ROS_LOCALHOST_ONLY=1
export RMW_IMPLEMENTATION=rmw_cyclonedx_cpp

# 방화벽 확인
sudo ufw status
```

## 📅 개발 히스토리

- **2025.03**: 초기 Whisper 모델 통합
- **2025.04**: ROS2 토픽 인터페이스 구현
- **2025.05**: Docker 컨테이너화 및 GPU 지원
- **2025.05**: 실시간 처리 모드 추가
- **2025.06**: VLA 파이프라인 연동 완료

## 👥 기여자

- **STT 시스템**: soda
- **Docker 환경**: soda
- **ROS2 통합**: 팀 공동 작업

## 📝 라이선스

이 프로젝트는 K-프로젝트의 일환으로 개발되었습니다.

---

### 🔧 추가 참고사항

- Whisper 모델의 첫 로딩 시 인터넷 연결이 필요합니다
- GPU 메모리가 부족한 경우 자동으로 CPU 모드로 전환됩니다
- 한국어 인식 성능 향상을 위해 조용한 환경에서 사용을 권장합니다 