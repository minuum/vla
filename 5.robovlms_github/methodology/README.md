# RoboVLMs Methodology Analysis

## Vision-Language Models (VLMs)

### 기본 구조
VLMs는 시각과 텍스트 정보를 처리하고 추론할 수 있는 멀티모달 대형 언어 모델입니다.

#### 수식 표현
```
l̂ = VLM(I, l_prompt)
```
- I: 입력 이미지
- l_prompt: 텍스트 프롬프트
- l̂: VLM이 생성한 텍스트 출력

#### 훈련 손실
```
L_VLM = CrossEntropy(l̂, l_target)
```

### 아키텍처 분류

#### 1. Encoder-Decoder 구조
- **구성**: 인코더(특징 추출) + 디코더(자동회귀 생성)
- **특징**: 인코더와 디코더 간 cross-attention을 통한 특징 융합
- **장점**: 입력 모달리티에 대한 상세한 이해
- **대표 모델**: Flamingo, OFA

#### 2. Decoder-Only 구조
- **구성**: 통합된 트랜스포머 프레임워크
- **특징**: 시각과 텍스트 토큰을 연결하여 self-attention으로 융합
- **장점**: 유연성과 확장성
- **대표 모델**: GPT-4V, LLaVA

## Vision-Language-Action Models (VLAs)

### 기본 정의
VLAs는 로봇 작업에 적용되는 일반화된 로봇 정책 π입니다.

#### 수식 표현
```
a_{t:t+L-1} = VLA(o_{t-H+1:t}, l_prompt)
```
- a_{t:t+L-1}: 예측된 7차원 액션 시퀀스
- L: 액션 시퀀스 길이
- H: 히스토리 관찰 길이

### 액션 전처리

#### 1. 액션 정규화
```python
# Quantile 기반 클램핑
a_i' = min(a_i^{99th}, max(a_i^{1st}, a_i))

# 정규화
ã_i = 2 × (a_i' - a_i^{1st}) / (a_i^{99th} - a_i^{1st}) - 1
```

#### 2. 액션 이산화 (Discrete Actions)
- 각 차원을 256개 빈으로 이산화
- 언어 토크나이저의 특수 토큰 위치 보호를 위해 오프셋 추가

### 액션 예측

#### 1. 연속 액션
```python
L_VLA = Σ_{i=t}^{t+L-1} [MSE(â_i,pose, ã_i,pose) + λ × BCE(a_i,gripper, ã_i,gripper)]
```

#### 2. 이산 액션
```python
L_VLA = Σ_{i=t}^{t+L-1} Σ_{j=1}^{7} CE([ACT]_i^j, ã_i^j)
```

## VLA 구조 분류

### 1. One-Step Models
현재 시간 단계 t의 관찰만 사용하여 미래 액션 시퀀스 예측

#### 연속 액션 모델
```python
[LRN] = VLM(o_t, l_prompt)
â_{t:t+L-1} = MLP([LRN])
```

#### 이산 액션 모델
```python
[ACT]_{t:t+L-1}^{1:7} = VLM(o_t, l_prompt)
```

### 2. Interleaved-Continuous-Action Models
관찰-액션 시퀀스를 교차 형식으로 처리

```python
O_t = ([OBS]_{t-H+1}, [LRN]), ..., ([OBS]_t, [LRN])
[LRN]_{t-H+1:t} = VLM(O_t)
â_{t:t+L-1} = MLP([LRN]_t)
```

### 3. Policy-Head-Continuous-Action Models
VLM이 단일 단계 멀티모달 표현을 제공하고, 별도의 정책 헤드가 히스토리 정보를 모델링

```python
o_t = ([OBS]_t, [LRN])
[LRN]_t = VLM(o_t, l_prompt)
a_{t:t+L-1} = h([LRN]_{t-H+1}, ..., [LRN]_t)
```

## RoboVLMs 프레임워크

### 핵심 특징
- **유연성**: 30줄 이내의 코드로 VLM을 VLA로 변환
- **통합성**: 다양한 VLM 백본과 VLA 구조 지원
- **확장성**: 새로운 VLM 통합 용이

### 구현 원리
1. VLM의 핵심 속성 설정
2. 멀티모달 특징 융합 메커니즘 정의
3. 액션 예측을 위한 추가 컴포넌트 통합

## 실험 설정

### 하이퍼파라미터
- **배치 크기**: 128
- **학습률**: 1e-4, 2e-5, 1e-5 중 선택
- **가중치 감쇠**: 0, 1e-1 중 선택
- **워밍업 비율**: 0.25 epoch

### 훈련 환경
- **하드웨어**: 4 x 8 A100 GPU 클러스터
- **훈련 에포크**: 5 epochs (CALVIN), 50K iterations (SimplerEnv)
- **체크포인트 선택**: 고정 에포크/반복 수 사용
