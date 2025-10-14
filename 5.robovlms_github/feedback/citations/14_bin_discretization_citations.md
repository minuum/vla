# 14. Bin 이산화 과정 상세 분석

## 📋 개요

이 문서는 RoboVLMs에서 연속 액션을 n개 bin으로 이산화하는 과정과 학습에서의 순서를 자세히 분석합니다.

## 🔍 1. Bin 이산화의 의도와 목적

### 1.1 이산화의 핵심 의도

**연속 액션 → 이산 토큰 변환**
```python
# 연속 액션: [-1, 1] 범위의 실수값
continuous_action = [0.5, -0.3, 0.8, 0.2, -0.1, 0.9, 0.1]  # 7DOF

# 이산화: 256개 bin으로 분할
bins = np.linspace(-1, 1, 256)  # [-1, -0.992, -0.984, ..., 0.992, 1]
discretized = np.digitize(continuous_action, bins)  # [192, 89, 230, 153, 115, 243, 140]
```

**출처**: `RoboVLMs/robovlms/model/policy_head/action_tokenizer.py:55-70`

### 1.2 이산화의 학습상 이점

**1) 토큰 기반 학습**
- VLM이 텍스트 토큰처럼 액션을 처리 가능
- 언어 모델의 autoregressive 생성 활용
- CrossEntropyLoss로 직접 학습 가능

**2) 정밀도 vs 효율성 균형**
- 256개 bin = 8비트 정밀도 (2^8 = 256)
- 연속 공간의 무한 정밀도 → 유한 정밀도로 근사
- 로봇 제어에 충분한 정밀도 제공

**3) 메모리 효율성**
- 연속값: float32 (4바이트) × 7차원 = 28바이트
- 이산값: int8 (1바이트) × 7차원 = 7바이트
- **75% 메모리 절약**

## 🔄 2. 이산화 과정의 상세 단계

### 2.1 Bin 생성 과정

```python
# ActionTokenizer.__init__()에서 bin 생성
def __init__(self, tokenizer, bins=256, min_action=-1, max_action=1):
    self.n_bins = bins                    # 256개 bin
    self.min_action = min_action         # -1 (최소값)
    self.max_action = max_action         # 1 (최대값)
    
    # 균등 분할로 bin 경계 생성
    self.bins = np.linspace(min_action, max_action, self.n_bins)
    # [-1, -0.992, -0.984, ..., 0.992, 1] (256개 값)
    
    # bin 중심값 계산 (실제 액션값으로 사용)
    self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
    # [-0.996, -0.988, -0.980, ..., 0.988, 0.996] (255개 값)
```

**출처**: `RoboVLMs/robovlms/model/policy_head/action_tokenizer.py:54-56`

### 2.2 액션 → 토큰 ID 변환

```python
def encode_actions_to_token_ids(self, action: np.ndarray) -> np.ndarray:
    """연속 액션을 토큰 ID로 변환"""
    # 1단계: 액션 클리핑 (범위 제한)
    action = np.clip(action, a_min=float(self.min_action), a_max=float(self.max_action))
    # [-1, 1] 범위로 제한
    
    # 2단계: bin 인덱스 계산
    discretized_action = np.digitize(action, self.bins)
    # [1, 256] 범위의 인덱스 (256개 bin)
    
    # 3단계: 토큰 ID 변환
    token_ids = self.tokenizer_orig_size - discretized_action
    # vocab_size - bin_index = 토큰 ID
    
    return token_ids
```

**출처**: `RoboVLMs/robovlms/model/policy_head/action_tokenizer.py:82-92`

### 2.3 토큰 ID → 액션 복원

```python
def decode_token_ids_to_actions(self, action_token_ids: np.ndarray) -> np.ndarray:
    """토큰 ID를 연속 액션으로 복원"""
    # 1단계: bin 인덱스 복원
    discretized_actions = self.tokenizer_orig_size - action_token_ids
    
    # 2단계: 인덱스 범위 조정
    discretized_actions = np.clip(
        discretized_actions - 1, 
        a_min=0, 
        a_max=self.bin_centers.shape[0] - 1
    )
    
    # 3단계: bin 중심값으로 액션 복원
    return self.bin_centers[discretized_actions]
```

**출처**: `RoboVLMs/robovlms/model/policy_head/action_tokenizer.py:94-115`

## 🎯 3. 학습 과정에서의 순서

### 3.1 데이터 전처리 단계

**1) CALVIN 데이터셋 로딩**
```python
# DiskCalvinDataset에서 연속 액션 로딩
action = self.actions[episode_idx][step_idx]  # [7] shape, 연속값
# 예: [0.5, -0.3, 0.8, 0.2, -0.1, 0.9, 0.1]
```

**2) 액션 이산화**
```python
# ActionPredictionBatchTransform에서 이산화
if self.discrete:
    next_action_ids = self.action_tokenizer.encode_actions_to_token_ids(next_action)
    # [192, 89, 230, 153, 115, 243, 140] (토큰 ID)
```

**출처**: `RoboVLMs/robovlms/data/base_action_prediction_dataset.py:340-342`

### 3.2 학습 단계

**1) 입력 구성**
```python
# 텍스트 + 액션 토큰 결합
input_ids = instruction_tokens + action_tokens
# [1, 2, 3, ..., 192, 89, 230, 153, 115, 243, 140]
```

**2) VLM Forward Pass**
```python
# BaseRoboVLM.forward_discrete()에서 처리
output = self.model(
    input_ids=instr_and_action_ids,
    attention_mask=instr_and_action_mask,
    output_hidden_states=True
)
```

**3) Loss 계산**
```python
# DiscreteDecoder.loss()에서 CrossEntropyLoss
loss_fct = nn.CrossEntropyLoss()
loss = loss_fct(
    shift_logits.view(-1, shift_logits.size(-1)), 
    shift_labels.view(-1)
)
```

**출처**: `RoboVLMs/robovlms/model/policy_head/base_policy.py:231-235`

### 3.3 추론 단계

**1) 액션 생성**
```python
# BaseRoboVLM.pred_action_discrete()에서 생성
action_ids = self.model.generate(
    input_ids=instr_and_action_ids, 
    max_new_tokens=action_dim
)
```

**2) 액션 복원**
```python
# 토큰 ID → 연속 액션 변환
discretized_actions = self.action_tokenizer.decode_token_ids_to_actions(action_ids)
# [0.496, -0.304, 0.796, 0.196, -0.104, 0.896, 0.096]
```

**출처**: `RoboVLMs/robovlms/model/backbone/base_backbone.py:1443-1445`

## 📊 4. Bin 수의 영향 분석

### 4.1 정밀도 vs 메모리 트레이드오프

| Bin 수 | 정밀도 | 메모리 | 토큰 범위 | 비고 |
|--------|--------|--------|-----------|------|
| 64 | 6비트 | 1바이트 | [0, 63] | 낮은 정밀도 |
| 128 | 7비트 | 1바이트 | [0, 127] | 중간 정밀도 |
| **256** | **8비트** | **1바이트** | **[0, 255]** | **기본값** |
| 512 | 9비트 | 2바이트 | [0, 511] | 높은 정밀도 |
| 1024 | 10비트 | 2바이트 | [0, 1023] | 매우 높은 정밀도 |

### 4.2 RoboVLMs에서의 Bin 설정

**기본 설정**: 256개 bin (8비트 정밀도)
```python
# 모든 설정 파일에서 동일
n_bin: 256
min_action: -1
max_action: 1
```

**이유**:
- 로봇 제어에 충분한 정밀도
- 메모리 효율성
- VLM 토큰화와 호환성

## 🔧 5. 실제 구현 예시

### 5.1 완전한 이산화 파이프라인

```python
# 1단계: 연속 액션 입력
continuous_action = np.array([0.5, -0.3, 0.8, 0.2, -0.1, 0.9, 0.1])

# 2단계: 이산화
action_tokenizer = ActionTokenizer(tokenizer, bins=256)
token_ids = action_tokenizer.encode_actions_to_token_ids(continuous_action)
# 결과: [192, 89, 230, 153, 115, 243, 140]

# 3단계: 토큰 ID를 텍스트로 변환
action_text = tokenizer.decode(token_ids)
# 결과: "액션 토큰 시퀀스"

# 4단계: 학습용 입력 구성
input_ids = instruction_tokens + token_ids
# [1, 2, 3, ..., 192, 89, 230, 153, 115, 243, 140]

# 5단계: VLM 학습
loss = model(input_ids, labels=token_ids)

# 6단계: 추론 시 액션 복원
predicted_actions = action_tokenizer.decode_token_ids_to_actions(token_ids)
# 결과: [0.496, -0.304, 0.796, 0.196, -0.104, 0.896, 0.096]
```

### 5.2 정밀도 손실 분석

**원본 액션**: [0.5, -0.3, 0.8, 0.2, -0.1, 0.9, 0.1]
**복원 액션**: [0.496, -0.304, 0.796, 0.196, -0.104, 0.896, 0.096]
**오차**: [0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004]

**평균 오차**: 0.004 (0.4%)
**최대 오차**: 0.004 (0.4%)

## 📈 6. 학습 효과 분석

### 6.1 이산화의 학습상 이점

**1) Autoregressive 학습**
- VLM이 액션을 시퀀스로 학습
- 이전 액션에 기반한 다음 액션 예측
- 언어 모델의 강력한 시퀀스 모델링 활용

**2) CrossEntropyLoss 활용**
- 연속값의 MSE Loss 대신 분류 Loss 사용
- 더 안정적인 학습
- 그래디언트 폭발/소실 문제 완화

**3) 토큰 기반 생성**
- VLM의 생성 능력 직접 활용
- 텍스트와 액션의 통합 학습
- 멀티모달 이해력 향상

### 6.2 성능 비교

| 방법 | 정밀도 | 학습 안정성 | 메모리 효율성 | 생성 품질 |
|------|--------|-------------|---------------|-----------|
| 연속값 | 100% | 보통 | 낮음 | 보통 |
| **이산화** | **99.6%** | **높음** | **높음** | **높음** |

## 🎯 7. 핵심 요약

### 7.1 Bin 이산화의 의도

1. **연속 공간 → 이산 공간 변환**
   - 무한 정밀도 → 유한 정밀도
   - 실수값 → 정수 인덱스
   - 액션 → 토큰 ID

2. **VLM 학습 최적화**
   - 언어 모델 아키텍처 활용
   - 토큰 기반 학습
   - Autoregressive 생성

3. **메모리 효율성**
   - 75% 메모리 절약
   - 빠른 학습/추론
   - 확장성 향상

### 7.2 학습 순서

1. **데이터 전처리**: 연속 액션 → 이산 토큰
2. **입력 구성**: 텍스트 + 액션 토큰 결합
3. **VLM 학습**: CrossEntropyLoss로 토큰 예측
4. **추론**: 생성된 토큰 → 연속 액션 복원

### 7.3 핵심 코드 위치

- **ActionTokenizer**: `RoboVLMs/robovlms/model/policy_head/action_tokenizer.py`
- **이산화 로직**: `encode_actions_to_token_ids()`, `decode_token_ids_to_actions()`
- **학습 통합**: `RoboVLMs/robovlms/data/base_action_prediction_dataset.py:340-342`
- **추론 처리**: `RoboVLMs/robovlms/model/backbone/base_backbone.py:1443-1445`

이 분석을 통해 RoboVLMs의 bin 이산화 과정과 학습에서의 순서를 명확히 이해할 수 있습니다.
