# Mobile VLA with RoboVLMs - 문서 목록

## 📚 문서 구조

이 디렉토리는 Mobile VLA 프로젝트의 모든 문서를 포함합니다.

### 🎯 **핵심 문서**

#### **1. 태스크 정의**
- **[MOBILE_VLA_TASK_DEFINITION.md](./MOBILE_VLA_TASK_DEFINITION.md)**
  - Mobile VLA 파인튜닝 태스크의 완전한 정의
  - 8가지 시나리오, 액션 공간, 입력 데이터 구조
  - 파인튜닝 대상 컴포넌트 및 학습 목표

#### **2. 사용자 가이드**
- **[MOBILE_VLA_GUIDE.md](./MOBILE_VLA_GUIDE.md)**
  - Mobile VLA 구현의 완전한 사용자 가이드
  - 환경 설정부터 Docker 배포까지 전체 과정
  - 태스크 정의, 학습, 추론, ROS2 통합

#### **3. 구현 요약**
- **[IMPLEMENTATION_SUMMARY.md](./IMPLEMENTATION_SUMMARY.md)**
  - Mobile VLA 구현의 기술적 요약
  - 완료된 작업 및 파일 구조
  - 실행 방법 및 테스트 결과

### 📋 **문서별 상세 내용**

#### **MOBILE_VLA_TASK_DEFINITION.md**
```
📖 내용:
├── 태스크 정의 (모바일 로봇 네비게이션)
├── 8가지 시나리오 (1박스/2박스, 세로/가로, 좌/우)
├── 액션 공간 정의 (2D 연속 액션)
├── 입력 데이터 구조 (시각/언어/로봇상태)
├── 데이터셋 특성 (72개 에피소드)
├── 파인튜닝 대상 컴포넌트
├── 학습 목표 및 성능 평가 지표
└── 실행 방법
```

#### **MOBILE_VLA_GUIDE.md**
```
📖 내용:
├── 태스크 정의 (요약)
├── 환경 설정 (하드웨어/소프트웨어)
├── 데이터셋 준비 (Mobile VLA 데이터셋)
├── 학습 실행 (PyTorch Lightning)
├── 추론 실행 (ROS2 통합)
├── Docker 배포 (컨테이너화)
└── 문제 해결 (트러블슈팅)
```

#### **IMPLEMENTATION_SUMMARY.md**
```
📖 내용:
├── 완료된 작업 (6개 주요 영역)
├── 파일 구조 (설정/데이터/학습/추론)
├── 실행 방법 (스크립트 및 Docker)
├── 테스트 결과 (통합 테스트)
└── 다음 단계 (실제 학습/평가)
```

### 🚀 **빠른 시작**

#### **1. 태스크 이해**
```bash
# 태스크 정의 확인
cat docs/MOBILE_VLA_TASK_DEFINITION.md
```

#### **2. 전체 가이드**
```bash
# 완전한 사용자 가이드
cat docs/MOBILE_VLA_GUIDE.md
```

#### **3. 구현 요약**
```bash
# 기술적 구현 요약
cat docs/IMPLEMENTATION_SUMMARY.md
```

### 📁 **관련 파일**

#### **설정 파일**
- `configs/mobile_vla/train_mobile_vla_full_ft.json` - 학습 설정
- `docker-compose-mobile-vla.yml` - Docker 설정

#### **구현 파일**
- `robovlms/data/mobile_vla_dataset.py` - 데이터셋 어댑터
- `train_mobile_vla.py` - 학습 스크립트
- `eval/mobile_vla/inference_wrapper.py` - 추론 래퍼

#### **실행 스크립트**
- `scripts/run_mobile_vla_train.sh` - 학습 실행
- `scripts/run_mobile_vla_inference.sh` - 추론 실행

### 🔗 **문서 간 연결**

```
MOBILE_VLA_TASK_DEFINITION.md (태스크 정의)
    ↓
MOBILE_VLA_GUIDE.md (사용자 가이드)
    ↓
IMPLEMENTATION_SUMMARY.md (구현 요약)
```

### 📊 **문서 통계**

| 문서 | 라인 수 | 주요 내용 |
|------|---------|-----------|
| MOBILE_VLA_TASK_DEFINITION.md | 303 | 태스크 정의, 액션 공간, 데이터셋 |
| MOBILE_VLA_GUIDE.md | 471 | 사용자 가이드, 환경 설정, 실행 방법 |
| IMPLEMENTATION_SUMMARY.md | 217 | 구현 요약, 완료된 작업 |
| README.md | 119 | 문서 목록 및 구조 |

---

*이 문서 목록은 Mobile VLA 프로젝트의 모든 문서를 체계적으로 정리합니다.*
