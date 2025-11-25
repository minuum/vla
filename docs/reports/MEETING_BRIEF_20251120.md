# 📅 2025-11-20 교수님 미팅 브리핑 및 향후 계획

**작성일:** 2025-11-25
**작성자:** Antigravity Agent
**관련:** 2025-11-20 미팅 피드백 및 이후 진행 상황

---

## 1. ✅ 현황 요약 (Status Summary)

### 1.1 데이터셋 수집 (Dataset Collection)
- **달성 현황**: 총 **484개** 수집 완료 (목표 500개 대비 96.8% 달성)
  - `1box_left`: 234개 (93.6%)
  - `1box_right`: 250개 (100% 완료)
- **이슈**: **Git LFS 대역폭 초과**로 인해 학습 서버에는 237개만 동기화됨.
  - 🚨 **영향**: 실제 학습은 전체 데이터의 절반으로만 수행됨. 성능 저하의 주원인일 가능성 높음.

### 1.2 모델 학습 및 성능 (Training & Performance)
- **Loss 추이**:
  - Epoch 0, 6에서 Val Loss **0.349**로 최고 성능 기록.
  - 이후 Epoch에서는 Loss가 튀거나 증가하는 **오버피팅(Overfitting)** 징후 관찰.
- **정성 평가**:
  - 장애물 회피 및 직진 경향성은 보이나, 일부 구간에서 불안정함.

### 1.3 환경 통제 (Environment Control)
- 커튼 설치 완료로 외부 광원 차단 및 조명 일관성 확보 성공.

---

## 2. 🧐 주요 의문점 및 검증 과제 (Key Questions)

### 2.1 Domain Gap: 7DOF Manipulator → 2DOF Mobile
- **의문**: RoboVLMs는 7자유도 로봇팔로 사전학습됨. 이를 2자유도 모바일 로봇(Velocity)에 적용했을 때, VLM이 추출하는 **Context Vector**가 유효한가?
- **검증 계획**: `analyze_context_vectors.py`를 실행하여 Mobile 데이터 입력 시 생성되는 벡터의 분포와 클러스터링을 시각화하여 분석.

### 2.2 데이터셋 양과 증강 (Data Augmentation)
- **의문**: 500개 수준의 데이터로 충분한 일반화가 가능한가?
- **계획**: 
  1. 현재 484개로 LFS 해결 후 재학습하여 베이스라인 성능 확인.
  2. 시뮬레이션 환경 구축 또는 기하학적 변환을 통한 데이터 증강(500 -> 5000개 목표) 검토.

---

## 3. 🚀 향후 계획 (Roadmap)

### Phase 1: 기반 확보 및 검증 (이번 주)
1.  **Git LFS 문제 해결 (최우선)**
    - Hugging Face Hub 등을 활용한 외부 스토리지로 데이터 이관.
    - 전체 484개 데이터셋 확보 및 동기화.
2.  **Context Vector 유효성 검증**
    - Pre-trained 모델의 Feature Extraction 능력 테스트.
3.  **Full Dataset 재학습**
    - 484개 데이터로 다시 학습 후 Loss Curve 및 Inference 성능 비교.

### Phase 2: 성능 고도화 (다음 주)
1.  **정량적 평가 지표 수립**
    - MSE (Mean Squared Error), Success Rate 측정 스크립트 작성.
    - 논문용 비교 테이블(Table 1, 2, 3) 데이터 생성.
2.  **데이터 증강 전략 수립 및 적용**
    - 부족한 시나리오(2box 등)에 대한 증강 실험.

### Phase 3: 논문 작성 및 마무리
1.  **실험 결과 정리**
2.  **논문 초안 작성** (Methodology: Event-Triggered VLA, Domain Adaptation)

---

## 4. Action Items (To-Do)

- [ ] `analyze_context_vectors.py` 실행 및 결과 분석 보고
- [ ] Git LFS 대안(Hugging Face) 구축 및 데이터 이관
- [ ] 전체 데이터셋 기반 재학습 실행 (`train_enhanced_model.py`)
- [ ] Inference 테스트 (`run_inference_test.py`) 결과 녹화 및 분석
