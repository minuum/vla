# Mobile VLA Project

## 📁 프로젝트 구조

```
Mobile_VLA/
├── core/                    # 핵심 코드
│   ├── *_core.py           # 핵심 기능
│   ├── data_core/          # 데이터 처리
│   └── train_core/         # 훈련 관련
├── models/                  # 모델 구현
│   ├── core/               # 핵심 모델
│   ├── experimental/       # 실험적 모델
│   ├── data/               # 데이터 분석
│   └── legacy/             # 레거시 코드
├── experimental/            # 실험적 기능
│   └── *_experimental.py   # 실험적 코드
├── results/                 # 결과 파일들
│   ├── *.json              # 성능 결과
│   ├── *.png               # 시각화
│   ├── *.pt                # 모델 체크포인트
│   └── *.log               # 로그 파일
├── docs/                    # 문서
│   ├── *.md                # 마크다운 문서
│   └── *.ipynb             # 노트북
├── legacy/                  # 레거시 데이터
└── README.md               # 이 파일
```

## 🎯 Core Components (핵심 구성요소)

### 🚀 핵심 기능
- `action_analysis_core.py` - 액션 분석
- `mobile_dataset_core.py` - 모바일 데이터셋
- `mobile_trainer_core.py` - 모바일 훈련기
- `mobile_trainer_simple_core.py` - 단순 훈련기
- `train_mobile_vla_core.py` - VLA 훈련 스크립트

### 📊 데이터 처리
- `data_core/` - 데이터 처리 모듈
- `train_core/` - 훈련 관련 모듈

## 🤖 Models (모델)

### 🎯 Core Models (핵심 모델)
- **훈련 스크립트**: CLIP+LSTM, LSTM 기반 훈련
- **인코더**: 모바일 이미지/텍스트 인코더
- **정책 헤드**: 모바일 정책 네트워크
- **유틸리티**: 추론, 최적화, 과적합 해결

### 🔬 Experimental Models (실험적 모델)
- **고급 모델**: 멀티모달, RoboVLMs 수정
- **분석 도구**: 정확도, 성능 분석
- **특수 기능**: Z축 처리, 차원 확인

### 📈 Data Analysis (데이터 분석)
- **데이터셋 분석**: 기본 분석, 증강 효과
- **로봇 증강**: 로봇 특화 증강 분석

### 📚 Legacy Code (레거시)
- **참고용**: Kosmos2 분석, RoboVLMs 스타일

## 🔬 Experimental Features (실험적 기능)

### 🧪 실험적 코드
- `example_usage_experimental.py` - 사용 예제
- 기타 실험적 기능들

## 📊 Results (결과)

### 📈 성능 결과
- **JSON 파일**: 훈련/평가 결과
- **PNG 파일**: 시각화 그래프
- **PT/PTH 파일**: 모델 체크포인트
- **LOG 파일**: 훈련 로그

### 📁 결과 디렉토리
- `simple_baseline_results/` - 기본 베이스라인
- `simple_clip_lstm_results_extended/` - CLIP+LSTM 확장
- `simple_lstm_results_extended/` - LSTM 확장

## 📚 Documentation (문서)

### 📖 분석 문서
- **성능 분석**: 다양한 성능 분석 보고서
- **모델 비교**: 모델 간 비교 분석
- **최적화 아이디어**: 성능 개선 방안
- **프로젝트 요약**: 전체 프로젝트 요약

### 📓 노트북
- **액션 예측**: 주요 분석 노트북
- **정리된 분석**: 정리된 분석 노트북

## 🗂️ Legacy Data (레거시 데이터)

### 📦 증강 데이터셋
- `augmented_dataset/` - 기본 증강 데이터
- `distance_aware_augmented_dataset/` - 거리 인식 증강

## 🚀 사용 가이드

### 🎯 빠른 시작
```bash
# 핵심 훈련
python core/train_mobile_vla_core.py

# 모델 훈련
python models/core/train_simple_clip_lstm_core.py

# 실험적 모델
python models/experimental/train_simplified_model.py
```

### 📊 성능 평가
```bash
# 정확도 분석
python models/experimental/accuracy_analysis_experimental.py

# 2D 평가
python models/experimental/accurate_2d_evaluation_eval.py
```

### 🔧 모델 사용
```python
from core.mobile_dataset_core import MobileDataset
from core.mobile_trainer_core import MobileTrainer
from models.core.mobile_image_encoder_core import MobileImageEncoder

# 데이터셋 초기화
dataset = MobileDataset()

# 훈련기 초기화
trainer = MobileTrainer()

# 인코더 초기화
encoder = MobileImageEncoder()
```

## 📋 파일 태그 시스템

### `_core.py`
- 핵심 기능을 담당하는 안정적인 코드
- 프로덕션 환경에서 사용 가능
- 지속적인 유지보수 대상

### `_experimental.py`
- 실험적 기능 및 연구용 코드
- 성능 검증이 필요한 코드
- 향후 core로 이동 가능

### `_data.py`
- 데이터 처리 및 분석 관련 코드
- 데이터셋 전처리 및 증강
- 통계 분석 및 시각화

### `_policy.py`
- 정책 네트워크 관련 코드
- 액션 예측 및 결정 로직
- 강화학습 정책 구현

### `_eval.py`
- 모델 평가 및 벤치마킹 코드
- 성능 측정 및 비교
- 테스트 스크립트

### `_legacy.py`
- 참고용 레거시 코드
- 더 이상 활발히 개발되지 않음
- 아카이브 목적

## 📈 성능 지표

### 주요 메트릭
- **MAE (Mean Absolute Error)**: 액션 예측 정확도
- **정확도**: 임계값별 성공률
- **R² 점수**: 모델 예측 능력
- **상관관계**: 예측-실제 간 상관관계

### 목표 성능
- **MAE**: < 0.1 (10cm 이내)
- **정확도 (0.3)**: > 80%
- **R² 점수**: > 0.7
- **상관관계**: > 0.8

## 🔄 마이그레이션 가이드

### Core로 이동 조건
- 충분한 테스트 완료
- 성능 검증 완료
- 문서화 완료
- 안정성 확인

### Legacy로 이동 조건
- 더 이상 사용되지 않음
- 대체 코드 존재
- 참고 가치만 남음

## 🛠️ 개발 가이드라인

### 코드 작성 규칙
1. 파일명에 적절한 태그 사용
2. 클래스명은 CamelCase
3. 함수명은 snake_case
4. 상세한 docstring 작성
5. 타입 힌트 사용

### 테스트 규칙
1. 새로운 기능은 반드시 테스트 작성
2. 실험적 코드는 별도 디렉토리에 배치
3. 성능 개선 시 벤치마크 실행
4. 문서화 필수

### 배포 규칙
1. Core 코드만 프로덕션 배포
2. Experimental 코드는 검증 후 이동
3. Legacy 코드는 참고용으로만 보관
4. 정기적인 코드 리뷰 및 정리

## 📝 프로젝트 요약

### 🎯 목표
Mobile VLA (Vision-Language-Action) 모델을 통한 로봇 제어 시스템 구현

### 🔧 주요 기능
- **Vision**: 이미지 인코딩 및 처리
- **Language**: 텍스트 명령 이해
- **Action**: 로봇 액션 예측 및 제어

### 📊 현재 상태
- 기본 모델 구현 완료
- 성능 최적화 진행 중
- 실험적 기능 개발 중

---

**📝 참고**: 이 프로젝트는 Mobile VLA 시스템의 모든 구성요소를 체계적으로 관리합니다. 새로운 기능 추가 시 적절한 태그를 사용하여 분류해주세요. 
