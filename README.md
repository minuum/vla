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

## 🏆 Latest Results: LoRA Fine-tuning

The following metrics represent the latest end-to-end training results using LoRA fine-tuning on the Kosmos-2 backbone.

### 📊 Training Metrics (Epoch 16)
- **Model**: Kosmos-2 + LoRA
- **LoRA Config**: Rank=32, Alpha=16, Dropout=0.1
- **Trainable Parameters**: 55.8M (Total: 1.7B)
- **Dataset**: 175 Train Episodes, 44 Validation Episodes
- **Training Loss**: **0.134**
- **Validation Loss**: **0.213**
- **Action Loss**: **0.213**

> [!NOTE]
> Training was validated up to Epoch 16. The low training and validation loss indicates successful convergence and effective learning of the mobile robot's action space.

## 📁 Key Directories
- `models/core/`: Core model training scripts
- `results/`: Checkpoints and performance logs
- `docs/`: Detailed analysis reports

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
