# RoboVLMs Experiments Analysis

## 실험 설계 개요

### 핵심 연구 질문별 실험
1. **Q1: 왜 VLA를 선호하는가?**
2. **Q2: 어떤 백본을 선택해야 하는가?**
3. **Q3: VLA 구조를 어떻게 공식화해야 하는가?**
4. **Q4: 언제 cross-embodiment 데이터를 활용해야 하는가?**

## 벤치마크 설정

### 1. CALVIN 벤치마크
- **설명**: 시뮬레이션 멀티태스크 테이블탑 조작
- **데이터**: 24K 인간 텔레오퍼레이션 시연, 언어 지시 포함
- **작업**: 34개 기본 기술 (블록 회전, 슬라이더 이동, 서랍 열기/닫기 등)
- **분할**: A, B, C, D (장면 설정별)
- **평가**: 5개 연속 작업 완료 성공률, 평균 달성 작업 수

### 2. SimplerEnv 벤치마크
- **설명**: 실제-시뮬 환경 스위트
- **환경**: WidowX+Bridge, Google Robot
- **작업**: 
  - WidowX+Bridge: Put Spoon on Towel, Put Carrot on Plate, Stack Block, Put Eggplant in Basket
  - Google Robot: Pick Coke Can, Move Near, Open/Close Drawer, Open Drawer & Place Apple

### 3. 실제 로봇 실험
- **플랫폼**: Kinova Gen-3 로봇 팔 + Robotiq 2F-85 그리퍼
- **카메라**: Kinect Azure (정적), RealSense D435i (손목)
- **작업**: 20개 작업, 5가지 설정
- **설정**: Simple, Unseen Distractor, Unseen Background, Unseen Object, Novel Skill Description

## 실험 결과

### Q1: 왜 VLA를 선호하는가?

#### CALVIN 성능 (ABCD → D)
| 방법 | VLA? | 1 | 2 | 3 | 4 | 5 | Avg. Len. |
|------|------|---|---|---|---|---|-----------|
| MCIL | ✖ | 0.373 | 0.027 | 0.002 | 0.000 | 0.000 | 0.40 |
| RT-1 | ✖ | 0.844 | 0.617 | 0.438 | 0.323 | 0.227 | 2.45 |
| GR-1 | ✔ | 0.949 | 0.896 | 0.844 | 0.789 | 0.731 | 4.21 |
| **KosMos P.H. (RoboVLMs)** | ✔ | **0.967** | **0.930** | **0.899** | **0.865** | **0.826** | **4.49** |

#### CALVIN 일반화 (ABC → D)
| 방법 | VLA? | 1 | 2 | 3 | 4 | 5 | Avg. Len. |
|------|------|---|---|---|---|---|-----------|
| GR-1 | ✔ | 0.854 | 0.712 | 0.596 | 0.497 | 0.401 | 3.06 |
| **KosMos P.H. (RoboVLMs)** | ✔ | **0.980** | **0.936** | **0.854** | **0.778** | **0.704** | **4.25** |

#### 실제 로봇 성능
- **Simple**: 75% 성공률
- **Unseen Distractor**: 60% 성공률
- **Unseen Background**: 50% 성공률
- **Unseen Object**: 55% 성공률
- **Novel Skill Description**: 33% 성공률

### Q2: 어떤 백본을 선택해야 하는가?

#### Vision-Language 사전 훈련의 영향
- **VL 사전 훈련 있음**: 4.49 Avg. Len. (ABCD), 4.25 Avg. Len. (ABC)
- **VL 사전 훈련 없음**: 2.70 Avg. Len. (ABCD), 0.56 Avg. Len. (ABC)

#### 다양한 VLM 백본 비교
- **KosMos**: 최고 성능
- **Flamingo**: 중간 성능
- **LLaVA**: Perceiver Resampler 추가 시 성능 향상

### Q3: VLA 구조를 어떻게 공식화해야 하는가?

#### 구조별 성능 비교
1. **Policy Head + Continuous Action**: 최고 성능
2. **Interleaved + Continuous Action**: 중간 성능
3. **One-Step + Discrete Action**: 낮은 성능

#### 일반화 및 데이터 효율성
- **Policy Head**: 히스토리 융합에 더 효과적이고 효율적
- **Interleaved**: 상대적으로 낮은 성능

### Q4: 언제 cross-embodiment 데이터를 활용해야 하는가?

#### 훈련 전략 비교
1. **Pre-train**: Cross-embodiment 데이터로 사전 훈련
2. **Post-train**: Cross-embodiment 사전 훈련 후 도메인 내 파인튜닝
3. **Finetune**: 도메인 내 데이터만 사용

#### 주요 발견사항
- **Pre-training**: Few-shot 학습에 도움 (17.2% 향상)
- **Post-training**: 전체 성능 향상 (52% vs 48% on Google Robot)
- **In-domain 데이터**: Cross-embodiment 데이터보다 효과적

## 데이터 효율성 분석

### 모델 크기별 성능 (3B ~ 9B)
- **3B**: 기본 성능
- **9B**: 향상된 성능, 특히 일반화 능력

### 데이터 스케일별 성능
- **10% 데이터 (0.1x ABCD)**: 1.38 Avg. Len.
- **표준 데이터 (ABCD)**: 4.49 Avg. Len.
- **500% 데이터 (5x ABCD)**: 4.51 Avg. Len.

## 자가 수정 능력

실제 로봇 실험에서 KosMos P.H. 모델이 자가 수정 능력을 보임:
- 첫 번째 시도 실패 시 자동으로 위치 조정
- 훈련 데이터에 없는 능력
- 다른 베이스라인 모델에서는 관찰되지 않음

## 실험의 한계

1. **아키텍처 제한**: 기존 VLM 구조 유지
2. **구조 분류 단순화**: 4가지 구조만 고려
3. **백본 제한**: 제한된 VLM 백본 세트
4. **실시간 배포**: 대형 모델의 실시간 제어 도전
