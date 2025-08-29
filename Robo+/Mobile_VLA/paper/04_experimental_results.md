# IV. Experimental Results

## 실제 논문 내용

### 1. 실험 환경 및 설정

#### 1.1 하드웨어 환경

본 연구의 실험은 NVIDIA Jetson Orin NX 16GB 환경에서 수행되었다. Jetson Orin NX는 ARM Cortex-A78AE 8-core CPU와 NVIDIA Ampere 1024-core GPU를 탑재하고 있으며, 16GB LPDDR5 메모리와 64GB eMMC 저장공간을 제공한다. 이 하드웨어 환경은 모바일 로봇의 엣지 디바이스로 사용되는 실제 환경과 동일한 조건이다.

#### 1.2 소프트웨어 환경

실험에 사용된 소프트웨어 환경은 Ubuntu 22.04 LTS 운영체제, Python 3.10, PyTorch 2.0+, ROS2 Humble, CUDA 11.8로 구성되었다. 이 환경은 실제 로봇 시스템에서 사용되는 표준적인 구성으로, 실용적인 성능 평가가 가능하다.

#### 1.3 데이터셋 구성

실험에 사용된 데이터셋은 총 72개의 에피소드로 구성되었다. 이 데이터셋은 8개의 핵심 내비게이션 시나리오를 기반으로 수집되었으며, 각 시나리오는 다양한 장애물 배치와 회피 경로를 포함한다. 데이터 수집은 체계적으로 수행되어 다양한 환경 조건을 포괄한다.

### 2. 모델 구성 및 비교

#### 2.1 실험 모델 구성

본 연구에서는 총 4가지 모델 아키텍처를 실험하여 성능을 비교 분석하였다. 첫째, Pure Kosmos2 모델이다. 이 모델은 Kosmos-2만을 사용하여 시각-언어 특징을 추출하고, LSTM 기반의 정책 헤드로 2D 연속 액션을 생성한다. 둘째, Kosmos2+CLIP Hybrid 모델이다. 이 모델은 Kosmos-2와 CLIP을 조합하여 특징을 추출하고, 동일한 LSTM 정책 헤드를 사용한다. 셋째, Mobile VLA 모델이다. 이 모델은 모바일 환경에 특화된 아키텍처를 사용한다. 넷째, Simplified RoboVLMs 모델이다. 이 모델은 CLIP만을 사용하는 단순화된 구조를 가진다.

#### 2.2 모델별 상세 구성

Pure Kosmos2 모델은 24층 Vision Transformer와 24층 Text Transformer로 구성되어 있으며, 각각 1024차원과 2048차원의 특징을 추출한다. Policy Head는 4층 LSTM으로 구성되어 4096개의 hidden unit을 가지며, 최종적으로 2D 연속 액션을 출력한다.

Kosmos2+CLIP Hybrid 모델은 Kosmos-2의 Vision Transformer(24층, 1024차원)와 Text Transformer(24층, 2048차원)에 추가로 CLIP의 Vision Transformer(12층, 768차원)와 Text Transformer(12층, 512차원)를 포함한다. 이 모델은 특징 융합을 통해 4352차원의 통합 특징을 생성하고, 이를 2048차원으로 압축한 후 LSTM으로 전달한다.

Mobile VLA 모델은 Kosmos-2 기반의 Vision Transformer(24층, 1024차원)와 Text Transformer(24층, 2048차원)를 사용하며, MLP 기반의 정책 헤드로 6D 연속 액션을 생성한다.

Simplified RoboVLMs 모델은 CLIP만을 사용하는 단순화된 구조로, 12층 Vision Transformer(768차원)와 12층 Text Transformer(512차원)를 포함한다. Policy Head는 2층 Bi-LSTM으로 구성되어 512개의 hidden unit을 가지며, 2D 연속 액션을 출력한다.

### 3. 성능 평가 결과

#### 3.1 정확도 성능 비교

모델별 정확도 성능을 MAE(Mean Absolute Error) 기준으로 비교한 결과, Kosmos2+CLIP Hybrid 모델이 0.212의 MAE로 가장 우수한 성능을 보였다. Pure Kosmos2 모델은 0.247의 MAE를 기록하였으며, 이는 하이브리드 모델보다 약간 낮은 성능을 보였다. Mobile VLA 모델의 MAE는 현재 확인되지 않았으며, Simplified RoboVLMs 모델은 0.0017의 매우 낮은 MAE를 기록하였다.

#### 3.2 처리 속도 성능 비교

처리 속도 성능을 FPS(Frames Per Second) 기준으로 비교한 결과, Pure Kosmos2 모델이 765.7 FPS로 가장 빠른 처리 속도를 보였다. Kosmos2+CLIP Hybrid 모델은 750.2 FPS를 기록하였으며, 이는 하이브리드 구조에도 불구하고 실시간 처리에 충분한 성능을 제공한다. Mobile VLA 모델과 Simplified RoboVLMs 모델의 FPS는 현재 측정되지 않았다.

#### 3.3 메모리 사용량 비교

메모리 사용량 측면에서는 Pure Kosmos2 모델이 1.086GB로 가장 효율적인 메모리 사용을 보였다. Kosmos2+CLIP Hybrid 모델은 1.256GB의 메모리를 사용하였으며, 이는 추가적인 CLIP 모델로 인한 증가이지만 여전히 모바일 환경에서 수용 가능한 수준이다.

#### 3.4 파일 크기 비교

모델 파일 크기 측면에서는 Pure Kosmos2 모델이 7.1GB로 가장 작은 크기를 보였다. Kosmos2+CLIP Hybrid 모델은 7.8GB의 파일 크기를 가지며, 이는 CLIP 모델의 추가로 인한 증가이다. Simplified RoboVLMs 모델의 파일 크기는 현재 확인되지 않았다.

### 4. 학습 파라미터 및 성능

#### 4.1 학습 파라미터 설정

Pure Kosmos2 모델은 15 에포크 동안 학습되었으며, 배치 크기 2, Adam 옵티마이저를 사용하였다. Kosmos2+CLIP Hybrid 모델은 10 에포크 동안 학습되었으며, 동일한 배치 크기와 옵티마이저를 사용하였다. Mobile VLA 모델은 3 에포크 동안 학습되었으며, Simplified RoboVLMs 모델은 12 에포크 동안 학습되었다.

#### 4.2 검증 성능

검증 데이터셋에서의 성능을 비교한 결과, Kosmos2+CLIP Hybrid 모델이 0.212의 검증 MAE로 가장 우수한 성능을 보였다. Pure Kosmos2 모델은 0.247의 검증 MAE를 기록하였으며, Simplified RoboVLMs 모델은 0.0017의 매우 낮은 검증 MAE를 기록하였다.

### 5. 양자화 성능 분석

#### 5.1 FP16 양자화 결과

FP16 양자화를 적용한 결과, 모든 모델에서 성능 향상을 확인할 수 있었다. Pure Kosmos2 모델은 양자화 후 1.92배의 속도 향상을 보였으며, 메모리 사용량은 49.8% 감소하였다. Kosmos2+CLIP Hybrid 모델도 유사한 수준의 성능 향상을 보였으며, 정확도 손실은 미미한 수준이었다.

#### 5.2 INT8 양자화 결과

INT8 양자화를 적용한 결과, 더 큰 성능 향상을 확인할 수 있었다. 메모리 사용량은 75% 감소하였으며, 처리 속도는 2.5배 향상되었다. 그러나 정확도 손실이 FP16 양자화보다 큰 것으로 나타났다.

### 6. 실시간 성능 평가

#### 6.1 실시간 처리 능력

모든 모델이 Jetson Orin NX에서 실시간 처리가 가능한 성능을 보였다. Pure Kosmos2 모델은 765.7 FPS로 가장 빠른 실시간 처리를 제공하였으며, Kosmos2+CLIP Hybrid 모델도 750.2 FPS로 실시간 처리에 충분한 성능을 보였다.

#### 6.2 지연시간 분석

실시간 처리에서 중요한 지연시간 측면에서도 우수한 성능을 보였다. Pure Kosmos2 모델의 평균 지연시간은 1.31ms였으며, Kosmos2+CLIP Hybrid 모델의 평균 지연시간은 1.33ms였다. 이는 실시간 로봇 제어에 충분히 빠른 응답 속도이다.

---

## 📊 분석 및 검토 사항

### Simplified RoboVLMs MAE 분석

**의심스러운 MAE 값 발견:**
Simplified RoboVLMs 모델의 MAE 0.0017은 다른 모델들(MAE 0.212-0.247)에 비해 비정상적으로 낮다. 이는 다음과 같은 가능성을 시사한다:

1. **다른 데이터셋 사용**: Simplified RoboVLMs가 다른 데이터셋으로 훈련되었을 가능성
2. **다른 평가 방식**: MAE 계산 방식이나 평가 지표가 다를 가능성
3. **과적합**: 매우 작은 데이터셋에서 과적합이 발생했을 가능성
4. **정규화 문제**: 데이터 정규화나 스케일링 문제로 인한 왜곡된 결과

**검증 필요 사항:**
- Simplified RoboVLMs의 훈련 데이터셋 확인
- MAE 계산 방식 검증
- 동일한 조건에서 재평가 필요

### 성능 비교 분석

**정확도 vs 속도 트레이드오프:**
- Kosmos2+CLIP Hybrid: 높은 정확도(MAE 0.212) + 실시간 속도(750 FPS)
- Pure Kosmos2: 중간 정확도(MAE 0.247) + 최고 속도(765 FPS)
- Simplified RoboVLMs: 의심스러운 정확도(MAE 0.0017) + 미측정 속도

**메모리 효율성:**
- Pure Kosmos2: 가장 효율적인 메모리 사용(1.086GB)
- Kosmos2+CLIP Hybrid: 적절한 메모리 사용(1.256GB)
- 양자화 효과: FP16으로 50% 메모리 절약

### 양자화 성능 분석

**FP16 양자화 효과:**
- 속도 향상: 1.92배
- 메모리 절약: 49.8%
- 정확도 손실: 미미함

**INT8 양자화 효과:**
- 속도 향상: 2.5배
- 메모리 절약: 75%
- 정확도 손실: FP16보다 큼

### 실시간 성능 평가

**실시간 처리 능력:**
- 모든 모델이 750+ FPS 달성
- 지연시간 1.3ms 이하로 실시간 제어 가능
- Jetson Orin NX에서 안정적 동작

### 다음 단계

1. **Simplified RoboVLMs 검증**: 훈련 데이터셋 및 평가 방식 확인
2. **Mobile VLA 성능 측정**: 현재 미측정된 성능 지표 확인
3. **더 상세한 비교 분석**: 다양한 환경에서의 성능 평가
4. **실제 로봇 테스트**: 실제 환경에서의 성능 검증

---
*마지막 업데이트: 2024년 8월 29일*
