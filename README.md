# 🤖 VLA (Vision-Language-Action) 프로젝트

**Vision-Language-Action 통합 로봇 시스템** - 컴퓨터 비전, 자연어 처리, 로봇 제어를 통합한 지능형 로봇 플랫폼

## 🏗️ **프로젝트 구조 (v2.0)**

```
vla/
├── 📁 core/                    # 핵심 시스템
│   ├── 🤖 robovlms/           # RoboVLMs 통합
│   ├── 🚀 ros/                # ROS2 시스템
│   ├── ⚙️ install/            # 설치 스크립트
│   └── 🧠 models/             # AI 모델들
├── 📚 docs/                    # 문서
│   ├── 📋 project/            # 프로젝트 개요
│   ├── 📈 progress/           # 진행상황
│   ├── 🎤 presentations/      # 발표 자료
│   └── 🤖 robotics/           # 로봇 관련
├── 🐳 docker/                  # Docker 환경
├── 📜 scripts/                 # 실행 스크립트
└── 📖 README.md               # 이 파일
```

## 🚀 **빠른 시작**

### **1. 환경 설정**
```bash
# Poetry 환경 활성화
poetry install
poetry shell

# ROS2 환경 설정
source /opt/ros/humble/setup.bash
```

### **2. 시스템 실행**
```bash
# 핵심 시스템 실행
./scripts/run_core_system.sh

# 모바일 VLA 데모
./scripts/run_mobile_vla_demo.sh

# RoboVLMs 테스트
./scripts/run_robovlms_docker.sh
```

### **3. Docker 환경**
```bash
# 모바일 VLA Docker 빌드
cd docker/
docker build -f Dockerfile.mobile-vla -t mobile-vla .

# 실행
docker run -it --gpus all mobile-vla
```

## 🎯 **주요 기능**

### **🧠 AI 모델**
- **Kosmos-2**: 멀티모달 이해 및 추론
- **PaliGemma**: 시각-언어 모델
- **Whisper**: 음성-텍스트 변환

### **🤖 로봇 제어**
- **ROS2**: 로봇 운영체제
- **Omni Controller**: 전방향 이동 제어
- **LiDAR**: 장애물 감지 및 회피

### **📱 모바일 플랫폼**
- **Jetson**: 엣지 AI 컴퓨팅
- **Camera**: 실시간 비전 처리
- **Voice**: 음성 명령 인식

## 📋 **시스템 요구사항**

- **OS**: Ubuntu 22.04 LTS
- **ROS**: ROS2 Humble
- **Python**: 3.10+
- **GPU**: NVIDIA Jetson Orin / RTX 4090+
- **Memory**: 16GB+ RAM

## 🔧 **설치 및 설정**

자세한 설치 가이드는 [docs/install/](docs/install/) 폴더를 참조하세요.

## 📚 **문서**

- **프로젝트 개요**: [docs/project/](docs/project/)
- **진행상황**: [docs/progress/](docs/progress/)
- **발표 자료**: [docs/presentations/](docs/presentations/)
- **로봇 가이드**: [docs/robotics/](docs/robotics/)

## 🤝 **기여하기**

1. 이슈 생성 또는 기존 이슈 확인
2. `cleanup/project-restructure-v2` 브랜치에서 작업
3. Pull Request 생성

## 📄 **라이선스**

Apache License 2.0

## 🔗 **관련 링크**

- [RoboVLMs](https://github.com/THUDM/RoboVLMs)
- [ROS2](https://docs.ros.org/en/humble/)
- [Kosmos-2](https://github.com/microsoft/unilm/tree/master/kosmos-2)

---

**버전**: v2.0 (Project Restructure)  
**최종 업데이트**: 2025년 9월 3일  
**상태**: 🟢 정리 완료 
