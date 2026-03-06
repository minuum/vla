# Phase 1-2: Omni-directional Target Navigation (Gray Basket)

## 1. 개요 및 목적 (Objective)
교수님 미팅 결과에 따라, 당장 복잡한 장애물 회피(옵스타클)로 넘어가기 전 중간 단계인 **Phase 1-2**를 신설합니다. 
목표는 타겟(회색 바스켓)이 카메라 FOV(Field of View) 내 중앙이 아닌 **어느 위치(좌, 우, 극좌, 극우)에 나타나든 해당 방향으로 정확히 로봇이 회전하여 찾아가는 타겟 지향 주행(Target-Driven Navigation)** 능력을 완벽히 확보하는 것입니다.

이 단계를 거치면 모델은 "진행 시간(프레임)"이 아닌 "현재 시야에 보이는 바스켓의 이미지 좌표"를 기반으로 액션을 결정(Grounding)하게 되며, 타이밍 암기(Causal Confusion) 문제가 완전히 파훼됩니다.

---

## 2. 데이터 수집 계획 (Data Collection Plan)

**핵심 타겟**: 오직 회색 바스켓 (갈색 화분 등 장애물 배제)

모델이 시각적 맥락에만 의존하도록 5가지 다양한 출발 각도에서 에피소드를 수집합니다. (총 250~300 에피소드 권장)

| 시나리오 위치 | 바스켓의 시야 내 초기 위치 | 초기 액션 (로봇 조작) | 수집 목표량 |
|:---:|:---|:---|:---:|
| **Far Left** | 카메라 시야의 극좌측 | 강한 좌회전 (Hard Left) | 50~60 eps |
| **Center Left** | 카메라 시야 좌측 중앙 | 부드러운 좌회전 (Soft Left) | 50~60 eps |
| **Center** | 정중앙 | 직진 (Forward) | 50~60 eps |
| **Center Right** | 카메라 시야 우측 중앙 | 부드러운 우회전 (Soft Right) | 50~60 eps |
| **Far Right** | 카메라 시야의 극우측 | 강한 우회전 (Hard Right) | 50~60 eps |

**데이터 수집 가이드(Teleoperation)**:
1. 시작 시점에 로봇과 바스켓의 상대 각도를 명확히 둔 채 녹화를 시작합니다.
2. 바스켓을 향해 로봇을 제비어(회전)하며 접근합니다.
3. 바스켓이 중앙에 오면 그대로 직진하여 목표 지점에서 정지 후 종료합니다.
4. 시작 지점과 바스켓 간의 거리를 다양화(1m, 1.5m, 2m 등)하여 스케일(크기)에 대한 강건성을 확보합니다.

---

## 3. 데이터셋 구성 계획 (Dataset Configuration)

- **Instruction 설계**: `"Navigate toward the gray basket."`
  - 단일 명령어로 통일합니다. 장애물을 언급하지 않으며, 오직 "회색 바스켓을 찾아가라"는 지시를 통해 모델이 픽셀에서 바스켓 특징 맵을 추출하는 것에만 집중하도록 유도합니다.

- **타이밍 암기 원천 차단 효과**:
  - 기존 v2는 항상 정해진 프레임에 장애물이 등장하여 회전했습니다. 
  - Phase 1-2는 프레임 0부터 강한 좌회전, 직진, 강한 우회전 등 액션이 랜덤하게 시작되므로, Frame Number와 Action 간의 상관관계(Spurious Correlation)가 완전히 끊어집니다.

- **기존 데이터 (basket_dataset_v2) 렌더링**:
  - 이전의 `basket_dataset_v2`는 고정된 회피 궤적(Timing Issue)을 학습하므로 **새로운 Phase 1-2 모델 학습 시 텐서에서 제외**하거나, 직진성이 강한 에피소드 극소수(Option)만 섞고 메인으로 사용하지 않습니다.

---

## 4. 학습 전략 (Training Strategy)

현행 최고 성능과 안정성을 증명한 **V3-EXP08 아키텍처를 그대로 유지**하여 학습 효율을 극대화합니다.

1. **모델 아키텍처**:
   - VLM Backbone: Kosmos-2 (Freeze VLM Core)
   - Adapter: LoRA (rank=32, alpha=64) On Attention Layers
   - Visual Resampler: Perceiver Resampler (64 Latents)
   - Temporal Encoding: LSTM (Window Size=8)
   - Action Head: 9-Class Discrete Classification (이산 액션)

2. **클래스 가중치 (Class Weighting)**:
   - 다양한 각도의 데이터를 수집했더라도, 주행 후반부에는 바스켓이 중앙에 오며 "직진(Forward)" 액션이 압도적으로 많아집니다.
   - 데이터로더에서 F, FL, FR, HL, HR 등의 빈도수를 계산 후 **실측값 역수 기반 가중치(Inverse Frequency Weighting)**를 적용하여 직진 편향을 막습니다.

3. **초매개변수 (Hyperparameters)**:
   - Learning Rate: `1e-5` (LoRA 미세조정에 최적화된 학습률 재사용)
   - Batch Size: `4` to `8`

---

## 5. 검증 계획 (Validation Plan)

- **오프라인 메트릭 (Offline Validation)**:
  - Val Loss 감소 확인 및 PM(Perfect Match) / DM(Direction Match) 타겟 85~95% (100% 도달 시 Overfitting 경계)
- **온라인 추론 (Real-world Inference)**:
  - (테스트 1) 로봇을 바스켓 45도 방향으로 둔 후, 바스켓을 감지하고 회전(Centering)하여 직진하는지 확인.
  - (테스트 2) 이동 도중 바스켓을 화면 가장자리로 몰래 이동시켰을 때(동적 환경), 로봇이 즉시 궤적을 수정(Reactive)하여 바스켓을 쫓아가는지 확인.
