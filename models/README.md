# Mobile VLA Models Directory

## 📁 디렉토리 구조

```
models/
├── core/                    # 핵심 모델 및 훈련 코드
├── experimental/            # 실험적 모델 및 분석 코드
├── data/                    # 데이터 처리 및 분석 코드
├── legacy/                  # 레거시 코드 (참고용)
└── README.md               # 이 파일
```

## 🎯 Core Models (핵심 모델)

### 🚀 훈련 스크립트
- `train_simple_clip_lstm_core.py` - CLIP + LSTM 기반 기본 훈련
- `train_simple_lstm_core.py` - LSTM 기반 훈련
- `enhanced_training_core.py` - 향상된 훈련 스크립트
- `improved_training_core.py` - 개선된 훈련 스크립트
- `no_augmentation_training_core.py` - 증강 없는 훈련
- `task_specific_training_core.py` - 태스크 특화 훈련
- `conservative_augmentation_training_core.py` - 보수적 증강 훈련
- `fix_shape_error_training_core.py` - 형태 오류 수정 훈련

### 🔧 인코더
- `mobile_image_encoder_core.py` - 모바일 이미지 인코더
- `korean_text_encoder_core.py` - 한국어 텍스트 인코더
- `mobile_policy_head_policy.py` - 모바일 정책 헤드

### 🛠️ 유틸리티
- `inference_core.py` - 추론 코드
- `hybrid_optimization_strategy_core.py` - 하이브리드 최적화 전략
- `overfitting_solution_core.py` - 과적합 해결책
- `set_token_core.py` - 토큰 설정

## 🔬 Experimental Models (실험적 모델)

### 🧪 실험적 모델
- `advanced_multimodal_model_experimental.py` - 고급 멀티모달 모델
- `fixed_robovlms_model_experimental.py` - 수정된 RoboVLMs 모델

### 📊 분석 및 평가
- `accuracy_analysis_experimental.py` - 정확도 분석
- `performance_summary_experimental.py` - 성능 요약
- `accurate_2d_evaluation_eval.py` - 정확한 2D 평가
- `simple_comparison_experimental.py` - 간단한 비교

### 🔍 특수 기능
- `z_axis_special_handling_experimental.py` - Z축 특별 처리
- `check_action_dimensions_experimental.py` - 액션 차원 확인
- `train_simplified_model.py` - 단순화된 모델 훈련

## 📊 Data Processing (데이터 처리)

### 📈 데이터 분석
- `dataset_analysis_data.py` - 데이터셋 분석
- `augmentation_analysis_data.py` - 증강 분석
- `augmentation_effectiveness_analysis_data.py` - 증강 효과 분석
- `robotics_augmentation_analysis_data.py` - 로봇 증강 분석

## 📚 Legacy Code (레거시 코드)

### 🔄 참고용 코드
- `kosmos2_analysis_legacy.py` - Kosmos2 분석 (참고용)
- `robovlms_style_single_image_model_legacy.py` - RoboVLMs 스타일 모델 (참고용)

## 🎯 사용 가이드

### 🚀 빠른 시작
```bash
# 기본 훈련
python models/core/train_simple_clip_lstm_core.py

# 실험적 모델 훈련
python models/experimental/train_simplified_model.py

# 데이터 분석
python models/data/dataset_analysis_data.py
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
from models.core.mobile_image_encoder_core import MobileImageEncoder
from models.core.korean_text_encoder_core import KoreanTextEncoder
from models.core.mobile_policy_head_policy import MobilePolicyHead

# 모델 초기화
image_encoder = MobileImageEncoder()
text_encoder = KoreanTextEncoder()
policy_head = MobilePolicyHead()
```

## 📋 파일 태그 설명

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

---

**📝 참고**: 이 디렉토리는 Mobile VLA 프로젝트의 모든 모델 관련 코드를 체계적으로 관리합니다. 새로운 기능 추가 시 적절한 태그를 사용하여 분류해주세요.
