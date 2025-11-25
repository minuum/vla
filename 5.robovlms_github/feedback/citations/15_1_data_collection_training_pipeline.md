# 15-1. VLM Fine-tuning과 LSTM Layer 학습: 데이터 수집 및 학습 파이프라인

## 📋 개요

이 문서는 RoboVLMs에서 데이터 수집부터 VLM Fine-tuning과 LSTM Layer 학습 과정까지를 일반적인 AI 학습 파이프라인 방식으로 자세히 설명합니다.

## ⚙️ 0. VLA 구조와 Action Space 설정

### 0.1 VLA (Vision-Language-Action) 모델 개요

**VLA 정의**:
```
at:t+L−1 = VLA(ot−H+1:t, lprompt)
```
- `at:t+L−1`: 예측된 7-DOF 액션 시퀀스 (Translation 3 + Rotation 3 + Gripper 1)
- `L`: 액션 시퀀스 길이 (action chunk size)
- `H`: 히스토리 관측 길이 (window size)
- `ot`: 현재 시간 t의 관측값 (시각 정보 + proprioceptive state)
- `lprompt`: 언어 프롬프트

**출처**: RoboVLMs 논문 Section B, Equation (4)

---

### 0.2 VLA 구조 분류 (4가지)

RoboVLMs 논문에서는 VLA를 **히스토리 정보 모델링 방식**과 **액션 공간**에 따라 4가지로 분류합니다.

#### **분류 기준**:
1. **히스토리 정보 처리**: One-step vs Interleaved vs Policy-Head
2. **액션 공간**: Continuous vs Discrete

**출처**: RoboVLMs 논문 Section C, Fig. 12

---

### 0.3 Action Space 개념

**`action_space`는 설정 파라미터**입니다. `continuous`와 `discrete`는 **같은 계층(선택지)에 있으며**, 실제 로봇 태스크의 특성에 따라 선택하는 **두 가지 액션 표현 방식**입니다.

```python
# Config 파일에서 action_space 설정
{
    "act_head": {
        "type": "LSTMDecoder",          # Policy Head 타입
        "action_space": "continuous",   # 선택: "continuous" 또는 "discrete"
        "hidden_size": 1024,
        "action_dim": 7,
        "down_sample": "none"
    }
}
```

**출처**: 
- `RoboVLMs/paligemma_config.json:33-44`
- `RoboVLMs/configs/k_project/ros2_automotive.json:40-61`

---

### 0.4 Action 전처리 과정

#### **0.4.1 Action Normalization (모든 액션 공간 공통)**

**1단계: Quantile 기반 Clipping**
```python
# 1st와 99th percentile 기반 clipping
ai′ = min(ai_99th, max(ai_1st, ai))
```
**출처**: RoboVLMs 논문 Equation (5)

**2단계: [-1, 1] 범위로 정규화**
```python
# 각 차원을 [-1, 1]로 정규화
ãi = 2 × (ai′ − ai_1st) / (ai_99th − ai_1st) − 1

# 정규화된 액션
ã = [ã1, ã2, ..., ã7]  # 각 차원 ∈ [-1, 1]
# 마지막 차원 (gripper): ∈ {-1, 1}
```
**출처**: RoboVLMs 논문 Equation (6)

**코드 구현**:
```python
# CALVIN 데이터셋 정규화
{
    "norm_action": true,
    "norm_min": -0.65,  # 1st percentile
    "norm_max": 0.65    # 99th percentile
}
```
**출처**: `RoboVLMs/configs/calvin_finetune/finetune_kosmos_cont-lstm-post_full-ft_text_vision_wd-0_ws-8_act-10.json:126-128`

---

#### **0.4.2 Action Discretization (Discrete 액션 공간 전용)**

**256개 Bin으로 균등 분할**:
```python
# 각 차원을 독립적으로 256개 bin으로 이산화
# bin width = (ai_99th - ai_1st) / 256

# 이산화된 액션: 7개 정수
a_discrete = [bin_idx1, bin_idx2, ..., bin_idx7]  # 각각 ∈ [0...255]

# 토큰 충돌 방지를 위한 offset 추가 (기본 10)
token_id = vocab_size - offset - bin_idx
```
**출처**: RoboVLMs 논문 "Action Discretization" 섹션

---

### 0.5 두 방식의 역할과 차이

#### **Continuous Action Space** (연속 액션 공간)
**역할**: 로봇의 액션을 **연속적인 실수 값**으로 직접 예측

**논문 수식**:
```
[LRN] = VLM(ot, lprompt)
ât:t+L−1 = MLP([LRN])  또는  h([LRN]t−H+1, ..., [LRN]t)
```
**출처**: RoboVLMs 논문 Equation (10), (14)

```python
# BaseRoboVLM 초기화
def __init__(self, ...):
    self.action_space = self.act_head_configs.get("action_space", "continuous")
    
    if self.action_space == "continuous":
        # 학습 가능한 액션 토큰 생성 (VLM이 이 토큰을 통해 액션 정보를 융합)
        self.action_token = nn.Parameter(torch.zeros(self.hidden_size))
        self.action_token.requires_grad_(True)
```

**출처**: `RoboVLMs/robovlms/model/backbone/base_backbone.py:115-127`

**처리 흐름**:
```python
# forward_action() - 액션 공간에 따라 분기
def forward_action(self, vision_x, lang_x, ...):
    action_space = self.act_head_configs.get("action_space", "continuous")
    
    if action_space == "discrete":
        return self.forward_discrete(...)  # 이산 액션 처리
    else:
        return self.forward_continuous(...)  # 연속 액션 처리 (기본값)
```

**출처**: `RoboVLMs/robovlms/model/backbone/base_backbone.py:1344-1382`

**손실 함수 (논문)**:
```python
# MSE Loss (처음 6개 차원) + BCE Loss (gripper 차원)
lVLA = Σ(MSE(âi,pose, ãi,pose) + λ * BCE(âi,gripper, ãi,gripper))
```
**출처**: RoboVLMs 논문 Equation (7)

**특징**:
- **출력 형태**: `(batch_size, seq_len, action_dim)` - 예: `[0.5, -0.3, 0.1, ..., 0.8]`
- **손실 함수**: MSE Loss (pose 6차원) + BCE Loss (gripper 1차원)
- **장점**: 정밀한 제어, 부드러운 동작
- **대표 모델**: ACT, BC-Z, MVP, R3M, VIMA, 3D Diffuser, RoboMamba, π0
- **사용 사례**: 로봇팔 정밀 조작, 연속 궤적 제어

**출처**: RoboVLMs 논문 "One-step continuous-action models"

---

#### **Discrete Action Space** (이산 액션 공간)
**역할**: 연속 액션을 **N개 bin으로 이산화**하여 **토큰 ID**로 예측 (VLM의 next-token prediction 방식 활용)

```python
# BaseRoboVLM 초기화
def __init__(self, ...):
    self.action_space = self.act_head_configs.get("action_space", "continuous")
    
    if self.action_space == "discrete":
        # ActionTokenizer 생성 (연속 값 → 토큰 ID 변환)
        self.action_tokenizer = ActionTokenizer(
            self.tokenizer,
            bins=self.act_head_configs["n_bin"],       # 기본 256개
            min_action=self.act_head_configs["min_action"],  # -1
            max_action=self.act_head_configs["max_action"],  # 1
        )
```

**출처**: `RoboVLMs/robovlms/model/backbone/base_backbone.py:115-122`

**ActionTokenizer의 역할**:
```python
class ActionTokenizer:
    def __init__(self, tokenizer, bins=256, min_action=-1, max_action=1):
        """연속 로봇 액션을 N개 bin으로 이산화하고 토큰 ID로 매핑"""
        self.n_bins = bins
        self.bins = np.linspace(min_action, max_action, self.n_bins)  # 균등 분할
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
        
        # 토큰 ID 범위 설정 (vocab의 마지막 부분 사용)
        self.tokenizer_orig_size = self.tokenizer.vocab_size - special_tokens - offset
    
    def encode_actions_to_token_ids(self, action: np.ndarray) -> np.ndarray:
        """연속 액션 → 토큰 ID 변환"""
        discretized_action = np.digitize(action, self.bins)
        return self.tokenizer_orig_size - discretized_action
    
    def decode_token_ids_to_actions(self, action_token_ids: np.ndarray) -> np.ndarray:
        """토큰 ID → 연속 액션 복원"""
        discretized_actions = self.tokenizer_orig_size - action_token_ids
        discretized_actions = np.clip(discretized_actions - 1, 0, len(self.bin_centers) - 1)
        return self.bin_centers[discretized_actions]
```

**출처**: `RoboVLMs/robovlms/model/policy_head/action_tokenizer.py:14-115`

**논문 수식**:
```
[ACT]^1:7_t:t+L−1 = VLM(ot, lprompt)
```
**출처**: RoboVLMs 논문 Equation (11)

**손실 함수 (논문)**:
```python
# Cross Entropy Loss (각 차원별로 독립적으로 계산)
lVLA = Σ Σ CE([ACT]^j_i, ã^j_i)
      i  j
# i: 시간 인덱스 (t:t+L-1)
# j: 액션 차원 인덱스 (1:7)
```
**출처**: RoboVLMs 논문 Equation (8)

**추론 시 De-tokenization**:
```python
# 토큰 ID → bin index → 연속 값 (bin center)
predicted_action = bin_centers[token_id_to_bin_idx]
```
**출처**: RoboVLMs 논문 "Discrete Actions" 섹션

**특징**:
- **출력 형태**: 토큰 ID 시퀀스 - 예: `[32145, 32089, 32178, ..., 32200]`
- **손실 함수**: CrossEntropyLoss (각 차원별 토큰 분류)
- **장점**: VLM의 언어 모델링 능력 활용, 메모리 효율적
- **대표 모델**: RT-1, RT-2, 3D-VLA, LAPA, OpenVLA, EmbodiedCOT
- **사용 사례**: 복잡한 multi-modal 융합, 언어-비전-액션 통합 학습

**출처**: RoboVLMs 논문 "One-step discrete-action models"

---

### 0.6 VLA 구조별 상세 분류

RoboVLMs 논문에서 제시한 4가지 VLA 구조를 **RoboVLMs 코드베이스와 대응**하여 설명합니다.

---

#### **0.6.1 One-Step-Continuous-Action Models**

**특징**: 
- 히스토리 길이 H = 1 (현재 관측값만 사용)
- MLP로 연속 액션 직접 예측

**논문 수식**:
```
ât:t+L−1 = VLA(ot, lprompt)
[LRN] = VLM(ot, lprompt)
ât:t+L−1 = MLP([LRN])
```
**출처**: RoboVLMs 논문 Equation (9), (10)

**대표 모델**: ACT, BC-Z, MVP, R3M, VIMA, 3D Diffuser, RoboMamba, π0

**RoboVLMs 구현**: 지원하지만 기본 설정 아님

---

#### **0.6.2 One-Step-Discrete-Action Models**

**특징**:
- 히스토리 길이 H = 1
- VLM의 next-token prediction으로 액션 토큰 생성

**논문 수식**:
```
[ACT]^1:7_t:t+L−1 = VLM(ot, lprompt)
```
**출처**: RoboVLMs 논문 Equation (11)

**대표 모델**: RT-1, RT-2, 3D-VLA, LAPA, **OpenVLA**, EmbodiedCOT

**RoboVLMs 구현**: `action_space: "discrete"` (코드에 존재하지만 실제 사용 안 함)

---

#### **0.6.3 Interleaved-Continuous-Action Models**

**특징**:
- VLM 백본 **내부**에서 히스토리 융합
- Decoder-only 구조에서만 가능
- 관측-액션 토큰 interleaved 형식

**논문 수식**:
```
Ot = ([OBS]t−H+1, [LRN]), ..., ([OBS]t, [LRN])
[LRN]t−H+1:t = VLM(Ot)
ât:t+L−1 = MLP([LRN]t)
```
**출처**: RoboVLMs 논문 Equation (12)

**대표 모델**: GR-1, OCTO, GR-2

**RoboVLMs 구현**: 지원하지 않음 (Policy-Head 방식 선호)

---

#### **0.6.4 Policy-Head-Continuous-Action Models** ⭐ **RoboVLMs의 선택**

**특징**:
- VLM은 단일 시간 단계의 multi-modal representation만 제공
- **Policy Head (LSTM/RNN/Transformer/Diffusion)가 히스토리 모델링 담당**
- Encoder-Decoder와 Decoder-only 구조 모두 가능

**논문 수식**:
```
ot = ([OBS]t, [LRN])
[LRN]t = VLM(ot, lprompt)
at:t+L−1 = h([LRN]t−H+1, ..., [LRN]t)
```
**출처**: RoboVLMs 논문 Equation (13), (14)

**대표 모델**: **RoboFlamingo**, RoboUniview, DeeRVLA, **RoboVLMs (Kosmos, PaliGemma, LLaVA 등)**

**RoboVLMs 구현**:
```json
{
    "act_head": {
        "type": "LSTMDecoder",           // Policy Head = LSTM
        "action_space": "continuous",     // Continuous 액션
        "with_history": true,             // 히스토리 사용
        "history_type": "post",           // LSTM이 히스토리 모델링
        "window_size": 1                  // VLM 입력은 단일 프레임
    }
}
```
**출처**: `RoboVLMs/configs/calvin_finetune/finetune_kosmos_cont-lstm-post_full-ft_text_vision_wd-0_ws-8_act-10.json:74-84`

**장점**:
1. **모듈성**: VLM과 Policy Head 독립적 학습
2. **유연성**: 다양한 VLM 백본 (Encoder-Decoder/Decoder-only) 사용 가능
3. **효율성**: VLM은 단일 프레임만 처리, LSTM이 temporal reasoning 담당
4. **성능**: CALVIN Avg. Len. **4.49** (전체 1위)

---

#### **0.6.5 VLA 구조 비교표**

| **구조** | **히스토리 위치** | **액션 공간** | **VLM 구조** | **대표 모델** | **RoboVLMs 사용** |
|---------|-----------------|--------------|------------|-------------|-----------------|
| One-Step Continuous | 없음 (H=1) | Continuous | Any | ACT, MVP | ❌ |
| One-Step Discrete | 없음 (H=1) | Discrete | Any | RT-2, OpenVLA | ❌ |
| Interleaved Continuous | VLM 내부 | Continuous | Decoder-only | GR-1, OCTO | ❌ |
| **Policy-Head Continuous** | **Policy Head** | **Continuous** | **Any** | **RoboVLMs, RoboFlamingo** | **✅ (표준)** |

**출처**: RoboVLMs 논문 Fig. 12, Section C

---

### 0.7 Policy Head에서의 처리

```python
# DiscreteDecoder - action_space에 따라 토큰 시퀀스 처리
class DiscreteDecoder(BasePolicyHead):
    def __init__(self, ..., action_space="continuous", ...):
        self.action_space = action_space  # continuous/discrete 둘 다 처리 가능
        
        # ActionTokenizer는 discrete일 때만 사용
        if action_space == "discrete":
            self.action_tokenizer = ActionTokenizer(tokenizer, bins=n_bin, ...)
    
    def process_token_sequence(self, tok_seq):
        """토큰 시퀀스 처리 - action_space에 따라 분기"""
        if self.action_space == "continuous":
            # 연속 액션: flatten dimension
            tok_seq = tok_seq.reshape(bs, seq_len, -1)
        
        elif self.action_space == "discrete":
            # 이산 액션: 그대로 pass (이미 토큰 ID 형태)
            pass
        
        return tok_seq
```

**출처**: `RoboVLMs/robovlms/model/policy_head/base_policy.py:173-221`

---

### 0.4 추론 시 분기 처리

```python
# inference() - 추론 시에도 action_space에 따라 분기
def inference(self, vision_x, lang_x, ...):
    prediction = {}
    action_space = self.act_head_configs.get("action_space", "continuous")
    
    if self.train_setup_configs["predict_action"]:
        if action_space == "discrete":
            # 이산 액션: autoregressive generation으로 토큰 ID 생성
            action = self.pred_action_discrete(lang_x, vision_x, ...)
            prediction["action"] = action
        else:
            # 연속 액션: forward_continuous로 직접 액션 값 예측
            prediction["action"] = self.forward_continuous(
                vision_x, lang_x, ..., mode="inference"
            )
    
    return prediction
```

**출처**: `RoboVLMs/robovlms/model/backbone/base_backbone.py:1454-1491`

---

### 0.8 비교 요약

#### **0.8.1 Continuous vs Discrete Action Space**

| **구분** | **Continuous** | **Discrete** |
|---------|----------------|--------------|
| **계층** | 같은 계층 (선택지) | 같은 계층 (선택지) |
| **설정 위치** | `act_head.action_space` | `act_head.action_space` |
| **출력 형태** | 연속 실수 값 `[0.5, -0.3, ...]` | 토큰 ID `[32145, 32089, ...]` |
| **손실 함수** | MSE + BCE (논문 Eq. 7) | CrossEntropyLoss (논문 Eq. 8) |
| **VLM 역할** | 특징 추출 + 액션 토큰 융합 | 특징 추출 + next-token prediction |
| **Policy Head** | MLP, LSTM (회귀) | DiscreteDecoder (분류) |
| **정밀도** | 연속 값 (매우 높음) | bin 크기에 따름 (256 bin → 0.0078 간격) |
| **메모리** | 많음 | 적음 (토큰 ID만 저장) |
| **대표 모델** | ACT, RoboVLMs, OCTO | RT-2, OpenVLA |
| **사용 사례** | 정밀 조작, 연속 궤적 | VLM next-token 활용 |

---

#### **0.8.2 VLA 구조별 비교 (논문 기준)**

| **구조** | **히스토리** | **액션** | **VLM 구조** | **성능** | **RoboVLMs** |
|---------|------------|---------|------------|---------|-------------|
| One-Step Cont. | H=1 | Continuous | Any | 낮음 | ❌ |
| One-Step Disc. | H=1 | Discrete | Any | 중간 | ❌ |
| Interleaved Cont. | VLM 내부 | Continuous | Decoder-only | 중상 | ❌ |
| **Policy-Head Cont.** | **Policy Head** | **Continuous** | **Any** | **최고 (4.49)** | **✅** |

**출처**: RoboVLMs 논문 Section C, README 성능 표

---

### 0.9 실제 Config 예시

**Continuous 설정**:
```json
{
    "act_head": {
        "type": "LSTMDecoder",
        "action_space": "continuous",
        "action_dim": 7,
        "hidden_size": 1024,
        "down_sample": "none"
    }
}
```
**출처**: `RoboVLMs/paligemma_config.json:33-44`

**Discrete 설정** (사용 시):
```json
{
    "act_head": {
        "type": "DiscreteDecoder",
        "action_space": "discrete",
        "action_dim": 7,
        "n_bin": 256,
        "min_action": -1,
        "max_action": 1
    }
}
```
**출처**: `RoboVLMs/README.md:253-268`

---

### 0.10 RoboVLMs에서 실제 사용하는 Action Space

#### **결론: RoboVLMs는 Policy-Head-Continuous-Action 구조를 표준으로 사용합니다**

**선택 이유**:
1. **VLM 유연성**: Encoder-Decoder (Kosmos, PaliGemma) / Decoder-only (LLaVA) 모두 가능
2. **모듈 분리**: VLM (특징 추출) + LSTM (히스토리 + 액션 예측) 독립 학습
3. **효율성**: VLM은 단일 프레임만 처리, LSTM이 temporal reasoning
4. **성능**: CALVIN Avg. Len. **4.49** (전체 1위)

**출처**: RoboVLMs 논문 Section C.3 "Policy-Head-Continuous-Action Models"

**전체 Config 파일 분석 결과** (총 13개 설정):
- **`continuous`**: 11개 (84.6%)
- **`down_sample`**: 2개 (15.4%)
- **`discrete`**: 0개 (0%)

**출처**: `RoboVLMs/configs/` 전체 검색 결과

---

#### **0.7.1 Kosmos 모델 Config (CALVIN 최고 성능 모델)**

**CALVIN Benchmark 성능**:
- **ABCD → D**: 5-task Avg. Len. **4.49** (전체 1위)
- **ABC → D**: 5-task Avg. Len. **4.25** (전체 1위)

**출처**: `RoboVLMs/README.md:113-136`

**모든 Kosmos Config에서 `continuous` 사용**:

```json
// 1. CALVIN Fine-tuning (기본)
{
    "robovlm_name": "RoboKosmos",
    "act_head": {
        "type": "LSTMDecoder",
        "action_space": "continuous",  // ✅ continuous 사용
        "action_dim": 7,
        "down_sample": "none",
        "with_history": true,
        "history_type": "post"
    }
}
```
**출처**: `RoboVLMs/configs/calvin_finetune/finetune_kosmos_cont-lstm-post_full-ft_text_vision_wd-0_ws-8_act-10.json:74-84`

```json
// 2. CALVIN Fine-tuning (Hand RGB 사용)
{
    "robovlm_name": "RoboKosmos",
    "act_head": {
        "type": "LSTMDecoder",
        "action_space": "continuous",  // ✅ continuous 사용
        "action_dim": 7,
        "down_sample": "none",
        "window_size": 1
    }
}
```
**출처**: `RoboVLMs/configs/calvin_finetune/finetune_kosmos_cont-lstm-post_full-ft_text_vision_wd-0_use-hand_ws-16_act-10.json:74-84`

```json
// 3. OXE Pretrain (Real-World 데이터)
{
    "robovlm_name": "RoboKosmos",
    "act_head": {
        "type": "LSTMDecoder",
        "action_space": "continuous",  // ✅ continuous 사용
        "action_dim": 7,
        "down_sample": "none"
    }
}
```
**출처**: `RoboVLMs/configs/oxe_training/finetune_kosmos_cont-lstm-post_full-ft_text_vision_wd-0_use-hand_ws-16_act-10_oxe_pretrain.json:74-84`

```json
// 4. RT/Bridge Fine-tuning (Real-World)
{
    "robovlm_name": "RoboKosmos",
    "act_head": {
        "type": "LSTMDecoder",
        "action_space": "continuous",  // ✅ continuous 사용
        "action_dim": 7
    }
}
```
**출처**: 
- `RoboVLMs/configs/oxe_training/finetune_kosmos_cont-lstm-post_full-ft_text_vision_wd-0_use-hand_ws-16_act-10_rt_finetune.json:74-84`
- `RoboVLMs/configs/oxe_training/finetune_kosmos_cont-lstm-post_full-ft_text_vision_wd-0_use-hand_ws-16_act-10_bridge_finetune.json:74-84`

```json
// 5. Mobile VLA (실제 로봇 네비게이션)
{
    "robovlm_name": "RoboKosmos",
    "act_head": {
        "type": "LSTMDecoder",
        "action_space": "continuous",  // ✅ continuous 사용
        "action_dim": 7,
        "down_sample": "none",
        "window_size": 16
    }
}
```
**출처**: `RoboVLMs/configs/oxe_training/finetune_kosmos_mobile_vla.json:66-77`

---

#### **0.7.2 다른 VLM 모델들도 모두 `continuous` 사용**

**PaliGemma**:
```json
{
    "robovlm_name": "RoboPaligemma",
    "act_head": {
        "type": "LSTMDecoder",
        "action_space": "continuous",  // ✅ continuous 사용
        "action_dim": 7,
        "down_sample": "none"
    }
}
```
**출처**: `RoboVLMs/configs/calvin_finetune/finetune_paligemma_cont-lstm-post_full-ft_text_vision_wd=0_ws-8_act-10.json:74-84`

**LLaVA**:
```json
{
    "robovlm_name": "RoboLLaVA",
    "act_head": {
        "type": "LSTMDecoder",
        "action_space": "continuous",  // ✅ continuous 사용
        "action_dim": 7,
        "down_sample": "none"
    }
}
```
**출처**: `RoboVLMs/configs/calvin_finetune/finetune_llava-mpt-7b_cont-lstm-post_ful_ft_wd=0_hist-8_act-10.json:78-85`

**Qwen-VL**:
```json
{
    "robovlm_name": "RoboQwen",
    "act_head": {
        "type": "LSTMDecoder",
        "action_space": "continuous",  // ✅ continuous 사용
        "action_dim": 7,
        "down_sample": "none"
    }
}
```
**출처**: `RoboVLMs/configs/calvin_finetune/finetune_qwen-vl-7b_cont-lstm-post_full_ft_text_vision_wd=0_ws-8_act-10.json:74-84`

**Moondream**:
```json
{
    "robovlm_name": "RoboMoondream",
    "act_head": {
        "type": "LSTMDecoder",
        "action_space": "continuous",  // ✅ continuous 사용
        "action_dim": 7,
        "down_sample": "none"
    }
}
```
**출처**: `RoboVLMs/configs/calvin_finetune/finetune_moondream_cont-all-lstm-post_full-ft_text_vision_wd=0_ws-8_act-10.json:74-84`

---

#### **0.7.3 예외: `down_sample` 사용 사례 (2개)**

**Uform** (경량 모델):
```json
{
    "robovlm_name": "RoboUform",
    "act_head": {
        "type": "LSTMDecoder",
        "action_space": "down_sample",  // ⚠️ down_sample 사용 (예외)
        "action_dim": 7,
        "down_sample": "pooling",       // pooling 적용
        "token_source": "all"
    }
}
```
**출처**: `RoboVLMs/configs/calvin_finetune/finetune_uform_cont-lstm-post_full-ft_text_vision_wd=0_ws-8_act-10.json:74-85`

**Flamingo**:
```json
{
    "robovlm_name": "RoboFlamingo",
    "act_head": {
        "type": "LSTMDecoder",
        "action_space": "down_sample",  // ⚠️ down_sample 사용 (예외)
        "action_dim": 7,
        "down_sample": "pooling"
    }
}
```
**출처**: `RoboVLMs/configs/calvin_finetune/finetune_flamingo_mpt_3b_ws-8_act-10_lstm_calvin.json:72-82`

---

#### **0.10.4 왜 RoboVLMs는 Policy-Head-Continuous를 선호하는가?**

**1. 구조적 장점 (논문 기준)**

**Policy-Head vs Interleaved**:
```
Policy-Head:
- VLM 입력: 단일 프레임 ([OBS]t, [LRN])
- 히스토리 융합: LSTM에서 처리
- 장점: VLM과 Policy 독립적 최적화

Interleaved:
- VLM 입력: 전체 히스토리 ([OBS]t−H+1, [LRN]), ..., ([OBS]t, [LRN])
- 히스토리 융합: VLM 내부 self-attention
- 단점: Decoder-only만 가능, VLM 계산량 증가
```
**출처**: RoboVLMs 논문 Section C.2, C.3

**2. VLM 백본 호환성**
- **Encoder-Decoder**: Kosmos, PaliGemma (Cross-attention 활용) ✅
- **Decoder-only**: LLaVA, Qwen-VL (Self-attention 활용) ✅
- Interleaved는 Decoder-only만 가능 ❌

**3. Continuous의 정밀도 우위**
- 로봇팔 미세 움직임 제어
- MSE + BCE Loss로 직관적 학습 (논문 Eq. 7)
- Discrete는 256 bin 제약 (정밀도 0.0078 간격)

**4. 실제 성능 검증 (CALVIN Benchmark)**
```
Policy-Head Continuous (RoboVLMs Kosmos): 4.49 ⭐ (1위)
Interleaved Continuous (GR-1):           4.21    (2위)
One-Step Discrete (OpenVLA 추정):       ~3.5    (추정)
```
**출처**: `RoboVLMs/README.md:113-136`

**5. `discrete`를 사용하지 않는 이유**
- 추가적인 tokenization/de-tokenization overhead
- Bin 개수에 따른 정밀도 제한
- VLM의 next-token prediction은 **언어 태스크**에 최적화
- 로봇 조작에는 **연속 값 회귀**가 더 효과적
- RoboVLMs 실험: 모든 config에서 discrete 미사용 (0/13)

---

#### **0.10.5 요약: RoboVLMs의 선택**

| **모델** | **VLA 구조** | **Action Space** | **Down Sample** | **Policy Head** | **VLM 구조** | **성능** |
|---------|------------|-----------------|----------------|----------------|------------|---------|
| **Kosmos** (전체) | Policy-Head | `continuous` | `none` | LSTMDecoder | Encoder-Decoder | **4.49 ⭐** |
| **PaliGemma** | Policy-Head | `continuous` | `none` | LSTMDecoder | Encoder-Decoder | 고성능 |
| **LLaVA** | Policy-Head | `continuous` | `none` | LSTMDecoder | Decoder-only | 고성능 |
| **Qwen-VL** | Policy-Head | `continuous` | `none` | LSTMDecoder | Decoder-only | 고성능 |
| **Moondream** | Policy-Head | `continuous` | `none` | LSTMDecoder | Decoder-only | 고성능 |
| **Uform** | Policy-Head | `down_sample` | `pooling` | LSTMDecoder | Decoder-only | 경량 |
| **Flamingo** | Policy-Head | `down_sample` | `pooling` | LSTMDecoder | Decoder-only | 기본 |

**핵심**: RoboVLMs는 **Policy-Head-Continuous-Action 구조 (VLM + LSTM + MSE/BCE Loss)**를 표준으로 사용하여 CALVIN 최고 성능(4.49)을 달성했습니다.

**논문 근거**:
- Section C.3: "Policy-head-continuous-action models include RoboFlamingo, RoboUniview, and DeeRVLA"
- Equation (13), (14): Policy Head가 히스토리 모델링 담당
- Fig. 12: 4가지 VLA 구조 비교

**출처**: RoboVLMs 논문 Section C, README.md 성능 표

---

## 🔍 1. Real-World 데이터 수집 과정

### 1.1 CALVIN 데이터셋의 Real-World 특성

**데이터 수집 환경**
```python
# CALVIN 데이터셋의 실제 로봇 환경
obs_config = DictConfig({
    "rgb_obs": ["rgb_static", "rgb_gripper"],    # 정적 카메라 + 그리퍼 카메라
    "depth_obs": [],                             # 깊이 정보 (사용 안함)
    "state_obs": ["robot_obs"],                  # 로봇 상태 정보
    "actions": ["rel_actions"],                    # 상대적 액션
    "language": ["language"],                     # 언어 명령
})
```

**출처**: `RoboVLMs/robovlms/data/calvin_dataset.py:63-71`

**Real-World 데이터 구성**
- **Franka Emika Panda 7-DOF 로봇팔**: 실제 로봇 하드웨어
- **다중 카메라 시스템**: 정적 카메라 + 그리퍼 카메라
- **실제 물리 환경**: 테이블, 물체, 조작 공간
- **다양한 태스크**: pick-and-place, navigation, manipulation
- **실제 로봇 조작**: 전문가가 직접 조작하여 데이터 수집

### 1.2 데이터 전처리 과정

**이미지 전처리**
```python
# CALVIN 데이터셋의 이미지 처리
def process_rgb(self, episode, observation_space, transforms, seq_idx=0, window_size=0):
    # RGB 이미지 로딩 및 전처리
    rgb_static = episode["rgb_static"]      # 정적 카메라 이미지
    rgb_gripper = episode["rgb_gripper"]    # 그리퍼 카메라 이미지
    
    # 이미지 정규화 및 리사이징
    transforms = [
        Resize((224, 224)),                 # 224x224로 리사이징
        RandomHorizontalFlip(p=0.1),        # 제한적 증강
        ColorJitter(brightness=0.1, contrast=0.1),
        Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                 std=[0.26862954, 0.26130258, 0.27577711])
    ]
```

**출처**: `RoboVLMs/robovlms/data/calvin_dataset.py:236-243`

**액션 정규화**
```python
# 액션 정규화 과정
def collater(self, sample):
    if self.norm_action:
        for s in sample:
            s["actions"] = normalize_action(
                s["actions"], 
                self.norm_min,    # -1
                self.norm_max,    # 1
                maintain_last=True
            )
    
    # 그리퍼 액션 이진화
    action_tensors[..., -1] = ((action_tensors[..., -1] + 1) // 2).float()
```

**출처**: `RoboVLMs/robovlms/data/calvin_dataset.py:823-868`

## 🎯 2. VLM Fine-tuning 과정

### 2.1 VLM 아키텍처 선택

**지원되는 VLM 모델들**
```python
# 다양한 VLM 백본 지원
vlm_configs = {
    "PaliGemmaForConditionalGeneration": "paligemma-3b-pt-224",
    "RoboFlamingo": "flamingo-3b",
    "RoboKosmos": "kosmos-2",
    "RoboUform": "uform-vl-14b",
    "RoboPaligemma": "paligemma-3b-pt-224"
}
```

**출처**: `RoboVLMs/README.md:280-284`

### 2.2 Fine-tuning 설정

**Full Fine-tuning (F-FT) 설정**
```python
# Full Fine-tuning 설정
train_setup = {
    "lora_enable": False,           # LoRA 비활성화
    "freeze_backbone": False,       # 백본 모델 동결 해제
    "freeze_mm_mlp_adapter": False, # 멀티모달 어댑터 동결 해제
    "train_vision": True,           # 비전 모델 학습
    "train_text_embedding": True,   # 텍스트 임베딩 학습
    "precision": "bf16",            # BFloat16 정밀도
    "gradient_checkpointing": True  # 그래디언트 체크포인팅
}
```

**출처**: `RoboVLMs/configs/calvin_finetune/finetune_kosmos_cont-lstm-post_full-ft_text_vision_wd-0_ws-8_act-10.json:41-62`

**LoRA Fine-tuning 설정**
```python
# LoRA Fine-tuning 설정
train_setup = {
    "lora_enable": True,            # LoRA 활성화
    "lora_r": 64,                   # LoRA rank
    "lora_alpha": 16,               # LoRA alpha
    "lora_dropout": 0.05,           # LoRA dropout
    "lora_bias": "none",            # LoRA bias
    "freeze_backbone": True,        # 백본 모델 동결
    "train_vision": False,          # 비전 모델 동결
}
```

**출처**: `RoboVLMs/README.md:244-248`

### 2.3 VLM Fine-tuning 코드

**BaseRoboVLM 초기화**
```python
class BaseRoboVLM(nn.Module):
    def __init__(
        self,
        configs,
        train_setup_configs,
        act_head_configs=None,
        window_size=None,
        **kwargs,
    ):
        super().__init__()
        
        # 1단계: VLM 백본 초기화
        self.model = AutoModelForCausalLM.from_pretrained(
            configs["pretrained_model_name_or_path"]
        )
        
        # 2단계: 비전 타워 초기화
        self.vision_tower = self.model.vision_tower
        
        # 3단계: 텍스트 타워 초기화
        self.text_tower = self.model.language_model
        
        # 4단계: 액션 헤드 초기화
        self.act_head = self._init_heads()
        
        # 5단계: 학습 가능한 파라미터 설정
        self._trainable_params_setup()
```

**출처**: `RoboVLMs/robovlms/model/backbone/base_backbone.py:34-174`

**VLM Forward Pass**
```python
def forward(
    self,
    vision_x: torch.Tensor,
    lang_x: torch.Tensor,
    attention_mask: torch.Tensor = None,
    action_labels: Tuple[torch.Tensor, torch.Tensor] = None,
    action_mask: torch.Tensor = None,
    **kwargs,
):
    # 1단계: 멀티모달 입력 융합
    multimodal_embeds = self.merge_multi_modal_input(
        input_embeds=self.word_embedding(lang_x),
        vision_x=vision_x,
        attention_mask=attention_mask
    )
    
    # 2단계: VLM Forward Pass
    output = self.model(
        inputs_embeds=multimodal_embeds,
        attention_mask=attention_mask,
        output_hidden_states=True
    )
    
    # 3단계: 액션 헤드 Forward Pass
    action_loss = self._forward_action_head(
        action_tokens=output.hidden_states[-1],
        action_labels=action_labels,
        action_mask=action_mask
    )
    
    return action_loss
```

**출처**: `RoboVLMs/robovlms/model/backbone/base_backbone.py:1261-1318`

## 🧠 3. LSTM Layer 학습 과정

### 3.1 LSTM Decoder 아키텍처

**LSTMDecoder 초기화**
```python
class LSTMDecoder(BasePolicyHead):
    def __init__(
        self,
        in_features=1024,
        hidden_size=1024,
        action_dim=7,
        num_layers=2,
        down_sample="none",
        window_size=1,
        fwd_pred_next_n=1,
        **kwargs,
    ):
        super().__init__()
        
        # 1단계: LSTM 초기화
        self.rnn = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # 2단계: 액션 예측 헤드
        self.actions = nn.Linear(hidden_size, (action_dim - 1) * fwd_pred_next_n)
        self.gripper = nn.Linear(hidden_size, fwd_pred_next_n)
        
        # 3단계: 다운샘플링 설정
        self.down_sample = down_sample
        if down_sample == "pooling":
            self.global_1d_pool = nn.AdaptiveAvgPool1d(1)
        
        # 4단계: 히스토리 관리
        self.history_memory = []
        self.hidden_state = None
```

**출처**: `RoboVLMs/robovlms/model/policy_head/base_policy.py:142-192`

### 3.2 LSTM Forward Pass

**LSTMDecoder.forward()**
```python
def forward(self, tok_seq, h_0=None, **kwargs):
    # 1단계: 다운샘플링 처리
    if self.down_sample == "none":
        tok_seq = rearrange(tok_seq, "b l n d-> b l (n d)")
    elif self.down_sample == "pooling":
        tok_seq = self.global_1d_pool(tok_seq.permute(0, 2, 1))
    
    # 2단계: 히스토리 관리
    if tok_seq.shape[1] == 1:
        self.history_memory.append(tok_seq)
        if len(self.history_memory) <= self.history_len:
            # 히스토리 길이 내에서 LSTM 처리
            x, h_n = self.rnn(tok_seq, self.hidden_state)
            self.hidden_state = h_n
        else:
            # 윈도우 슬라이딩
            for _ in range(len(self.history_memory) - self.history_len):
                self.history_memory.pop(0)
            hist_feature = torch.cat(self.history_memory, dim=1)
            self.hidden_state = None
            x, h_n = self.rnn(hist_feature, self.hidden_state)
    else:
        # 배치 처리
        x, h_n = self.rnn(tok_seq, h_0)
        self.hidden_state = h_n
    
    # 3단계: 액션 예측
    actions = self.actions(x)      # 팔 액션 (6-DOF)
    gripper = self.gripper(x)      # 그리퍼 액션 (1-DOF)
    
    # 4단계: 출력 형태 조정
    actions = rearrange(actions, "b l (n d) -> b l n d", n=self.fwd_pred_next_n)
    gripper = rearrange(gripper, "b l (n d) -> b l n d", n=self.fwd_pred_next_n)
    
    return actions, gripper
```

**출처**: `RoboVLMs/robovlms/model/policy_head/base_policy.py:223-224`

### 3.3 LSTM Loss 계산

**LSTMDecoder.loss()**
```python
def loss(self, pred_action_logits, labels, attention_mask=None):
    # 1단계: 라벨 분리
    arm_action_labels, gripper_action_labels = labels
    
    # 2단계: 팔 액션 Loss 계산 (MSE Loss)
    arm_action_pred = pred_action_logits[..., :-1]
    loss_arm = F.mse_loss(arm_action_pred, arm_action_labels)
    
    # 3단계: 그리퍼 액션 Loss 계산 (BCE Loss)
    gripper_action_pred = pred_action_logits[..., -1]
    loss_gripper = F.binary_cross_entropy_with_logits(
        gripper_action_pred, 
        gripper_action_labels
    )
    
    # 4단계: 그리퍼 정확도 계산
    gripper_discrete_pred = (gripper_action_pred > 0).float()
    gripper_acc = (gripper_discrete_pred == gripper_action_labels).float().mean()
    
    return {
        "loss_arm": loss_arm,
        "loss_gripper": loss_gripper,
        "acc_gripper": gripper_acc
    }
```

**출처**: `RoboVLMs/robovlms/model/policy_head/base_policy.py:226-281`

## 🔄 4. 전체 학습 파이프라인

### 4.1 학습 데이터 로딩

**DiskCalvinDataset - 데이터 로딩**
```python
class DiskCalvinDataset(BaseCalvinDataset):
    def __getitem__(self, idx: Union[int, Tuple[int, int]]) -> Dict:
        # 1단계: 에피소드 로딩
        if isinstance(idx, int):
            episode = self._load_episode(idx, self.window_size)
        
        # 2단계: 이미지 처리
        image_seq = self.process_rgb(
            episode, 
            self.observation_space, 
            self.transforms
        )
        
        # 3단계: 액션 처리
        action_seq = episode["rel_actions"]
        
        # 4단계: 언어 처리
        task_description = episode["language"]["ann"][0]
        
        return {
            "image": image_seq,
            "action": action_seq,
            "task": task_description,
            "episode_mask": episode_mask
        }
```

**출처**: `RoboVLMs/robovlms/data/calvin_dataset.py:428-532`

### 4.2 학습 스텝

**BaseTrainer.training_step()**
```python
def training_step(self, batch, batch_idx):
    # 1단계: 배치 데이터 추출
    (rgb, hand_rgb, attention_mask, language, text_mask,
     arm_action, gripper_action, instr_and_action_ids,
     instr_and_action_labels, instr_and_action_mask) = self._process_batch(batch)
    
    # 2단계: 모델 Forward Pass
    prediction = self.model.forward(
        rgb,                    # 비전 입력
        language,               # 언어 입력
        attention_mask=text_mask,
        action_labels=(arm_action, gripper_action),
        action_mask=chunk_mask,
        instr_and_action_ids=instr_and_action_ids,
        instr_and_action_labels=instr_and_action_labels,
        instr_and_action_mask=instr_and_action_mask
    )
    
    # 3단계: Loss 계산
    output = self._get_loss(prediction)
    
    return output
```

**출처**: `RoboVLMs/robovlms/train/base_trainer.py:565-621`

### 4.3 Loss 계산

**BaseTrainer._get_loss()**
```python
def _get_loss(self, prediction):
    loss = {}
    total_loss = 0
    
    # 1단계: 팔 액션 Loss
    if "loss_arm" in prediction:
        loss_arm = prediction["loss_arm"]
        loss["loss_arm"] = loss_arm
        total_loss += loss_arm
    
    # 2단계: 그리퍼 액션 Loss
    if "loss_gripper" in prediction:
        loss_gripper = prediction["loss_gripper"]
        loss["loss_gripper"] = loss_gripper
        total_loss += self.arm_gripper_loss_ratio * loss_gripper
    
    # 3단계: VL Co-training Loss (선택적)
    if "loss_vl" in prediction:
        loss_vl = prediction["loss_vl"]
        loss["loss_vl"] = loss_vl
        total_loss += self.vl_cotrain_ratio * loss_vl
    
    # 4단계: 총 Loss
    loss["loss"] = total_loss
    
    return loss
```

**출처**: `RoboVLMs/robovlms/train/base_trainer.py:386-456`

### 4.4 Optimizer 설정

**BaseTrainer.configure_optimizers()**
```python
def configure_optimizers(self):
    # 1단계: 학습 가능한 파라미터 그룹화
    params = self.get_grouped_params(self.model)
    
    # 2단계: Optimizer 초기화
    if self.configs["optimizer"] == "adam":
        optimizer = torch.optim.Adam(
            params,
            lr=self.configs["learning_rate"],
            weight_decay=self.configs["weight_decay"]
        )
    elif self.configs["optimizer"] == "adamw":
        optimizer = torch.optim.AdamW(
            params,
            lr=self.configs["learning_rate"],
            weight_decay=self.configs["weight_decay"]
        )
    
    # 3단계: Learning Rate Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=self.configs["max_epochs"],
        eta_min=self.configs["learning_rate"] * self.configs["min_lr_scale"]
    )
    
    return [optimizer], [scheduler]
```

**출처**: `RoboVLMs/robovlms/train/base_trainer.py:716-722`

## 🔧 5. 실제 FT 코드와 LSTM Layer 학습 코드

### 5.1 VLM Fine-tuning 코드

**BaseRoboVLM._trainable_params_setup() - 파라미터 동결 설정**
```python
def _trainable_params_setup(self):
    model = self.model  # 백본 VLM 모델 (PaliGemma, Kosmos, LLaVA 등)
    
    # 1단계: 백본 모델 동결 설정
    if self.train_setup_configs["freeze_backbone"]:
        model.requires_grad_(False)  # 전체 모델 동결
    else:
        if self.train_setup_configs.get("train_decoder_layers", -1) == -1:
            model.requires_grad_(True)  # 전체 모델 학습
        else:
            # 마지막 N개 레이어만 학습
            model.requires_grad_(False)
            for layer in self.text_tower.layers[-self.train_setup_configs["train_decoder_layers"]:]:
                layer.requires_grad_(True)
    
    # 2단계: 비전 인코더 동결 설정
    # vision_tower: VLM의 비전 인코더 (CLIP, SigLIP 등)
    if self.train_setup_configs.get("train_vision", False):
        self.vision_tower.requires_grad_(True)
    else:
        self.vision_tower.requires_grad_(False)
    
    # 3단계: LoRA 설정
    if self.train_setup_configs["lora_enable"]:
        # LoRA 파라미터만 학습 가능하도록 설정
        pass
```

**vision_tower와 text_tower 설명**:
- **`vision_tower`**: VLM의 비전 인코더 부분 (이미지 → 특징 벡터)
  - PaliGemma: `model.vision_tower` (SigLIP 기반)
  - Kosmos: `model.vision_model` (CLIP 기반)
  - LLaVA: `model.get_vision_tower()` (CLIP 기반)
  - Flamingo: `self.vision_encoder` (CLIP 기반)

- **`text_tower`**: VLM의 텍스트/언어 모델 부분 (텍스트 → 특징 벡터)
  - PaliGemma: `model.language_model.model` (Gemma Decoder)
  - Kosmos: `model.text_model.model` (Decoder-only Transformer)
  - LLaVA: `model.transformer` (GPT-style Transformer)
  - Flamingo: `self.model` (언어 모델 전체)

**백본별 구현 예시**:
```python
# RoboPaligemma (robopaligemma.py:19-24)
@property
def text_tower(self):
    return self.model.language_model.model  # Gemma Decoder

@property
def vision_tower(self):
    return self.model.vision_tower  # SigLIP

# RoboKosMos (robokosmos.py:16-21)
@property
def text_tower(self):
    return self.model.text_model.model  # Transformer Decoder

@property
def vision_tower(self):
    return self.model.vision_model  # CLIP Vision

# RoboLLaVA (robollava.py:19-24)
@property
def text_tower(self):
    return self.model.transformer  # GPT Transformer

@property
def vision_tower(self):
    return self.model.get_vision_tower()  # CLIP Vision
```

**Kosmos2Processor 공식 문서 근거**:

Hugging Face 공식 문서에 따르면, Kosmos-2는 다음과 같이 구성됩니다:

```python
class transformers.Kosmos2Processor(
    image_processor,  # CLIPImageProcessor
    tokenizer,        # XLMRobertaTokenizerFast
    num_patch_index_tokens = 1024,
    **kwargs
)
```

**Parameters**:
- **image_processor** (`CLIPImageProcessor`) — An instance of `CLIPImageProcessor`. The image processor is a required input.
- **tokenizer** (`XLMRobertaTokenizerFast`) — An instance of `['XLMRobertaTokenizerFast']`. The tokenizer is a required input.

> "Constructs an KOSMOS-2 processor which wraps a KOSMOS-2 image processor and a KOSMOS-2 tokenizer into a single processor."

> "Kosmos2Processor offers all the functionalities of **CLIPImageProcessor** and some functionalities of **XLMRobertaTokenizerFast**."

이것이 Kosmos-2의 `vision_tower`가 CLIP 기반이고, `text_tower`가 XLM-Roberta 기반 Transformer인 이유입니다.

**출처**: 
- [Hugging Face KOSMOS-2 Documentation](https://huggingface.co/docs/transformers/en/model_doc/kosmos-2)
- `RoboVLMs/robovlms/model/backbone/base_backbone.py:470-512`
- `RoboVLMs/robovlms/model/backbone/robopaligemma.py:19-24`
- `RoboVLMs/robovlms/model/backbone/robokosmos.py:16-21`
- `RoboVLMs/robovlms/model/backbone/robollava.py:19-24`
- `RoboVLMs/robovlms/model/backbone/roboflamingo.py:35-40`

### 5.2 LSTM Layer 학습 코드

**LSTM 학습 루프 예시**
```python
# LSTM 학습 루프 (base_policy.py:625-642)
net = LSTMDecoder(
    in_features=1024,
    action_dim=7,
    down_sample="pooling",
    latent=1,
    fwd_pred_next_n=2,
    window_size=12,
)

optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
bs = 5
window_size = 12
text_len = 8
tokens = torch.randn(bs, window_size, text_len, 1024)
labels = (torch.randn(bs, window_size, 2, 6), torch.ones(bs, window_size, 2))
att_mask = torch.ones(bs, window_size, 2)

for i in range(10000):
    # Forward Pass
    actions, gripper = net(tokens)
    pred_action_logitss = torch.cat([actions, gripper.unsqueeze(-1)], dim=-1)
    
    # Loss 계산
    optimizer.zero_grad()
    loss = net.loss(pred_action_logitss, labels, att_mask)
    
    # Backward Pass
    loss_arm = loss["loss_arm"]
    loss_gripper = loss["loss_gripper"]
    acc_gripper = loss["acc_gripper"]
    loss_act = loss_arm + 0.01 * loss_gripper
    loss_act.backward()
    optimizer.step()
    
    print("iter: {}, loss: {} gripper: {} acc: {}".format(
        i, loss_act.item(), loss_gripper.item(), acc_gripper
    ))
```

**출처**: `RoboVLMs/robovlms/model/policy_head/base_policy.py:625-642`

### 5.3 Loss 계산 함수

**calculate_vl_cross_entropy() - Vision-Language Cross Entropy**
```python
def calculate_vl_cross_entropy(logits, labels, mask=None):
    # 1단계: 시퀀스 시프트
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # 2단계: Loss 계산
    if mask is None:
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, logits.shape[-1]),
            shift_labels.view(-1),
        )
    else:
        # 마스킹된 Loss 계산
        loss_fct = nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(
            shift_logits.view(-1, logits.shape[-1]),
            shift_labels.view(-1),
        )
        # 마스크 적용
        mask = mask[..., 1:].contiguous()
        loss = loss * mask.reshape(-1)
        loss = loss.mean()
    
    return loss
```

**출처**: `RoboVLMs/robovlms/train/loss.py:5-28`

### 5.4 설정 파일 예시

**CALVIN Fine-tuning 설정**
```json
{
  "train_setup": {
    "precision": "bf16",
    "predict_action": true,
    "predict_forward": false,
    "predict_caption": false,
    "train_vision": true,
    "freeze_backbone": false,
    "freeze_mm_mlp_adapter": false,
    "lora_enable": false,
    "train_text_embedding": true
  },
  "act_head": {
    "type": "LSTMDecoder",
    "hidden_size": 1024,
    "action_dim": 7,
    "down_sample": "none",
    "latent": 1,
    "fwd_pred_next_n": 1,
    "window_size": 1,
    "action_space": "continuous"
  }
}
```

**출처**: `RoboVLMs/README.md:228-267`

## 📊 6. 학습 흐름 요약

### 6.1 전체 학습 파이프라인

```
[데이터 로딩]
    ↓
[이미지 전처리] ← RGB 이미지 (224x224)
    ↓
[액션 정규화] ← 7-DOF 액션 (-1 to 1)
    ↓
[VLM Forward Pass] ← 멀티모달 융합
    ↓
[LSTM Forward Pass] ← 시퀀스 처리
    ↓
[Loss 계산] ← MSE (팔) + BCE (그리퍼)
    ↓
[Backward Pass] ← 그래디언트 계산
    ↓
[Optimizer Step] ← 파라미터 업데이트
```

### 6.2 주요 하이퍼파라미터

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| **Learning Rate** | 1e-4 ~ 2e-5 | 학습률 |
| **Batch Size** | 2 ~ 8 | 배치 크기 |
| **Window Size** | 8 ~ 16 | 히스토리 길이 |
| **Hidden Size** | 1024 | LSTM 은닉 차원 |
| **Action Dim** | 7 | 액션 차원 (6-DOF + 그리퍼) |
| **Precision** | bf16 | BFloat16 정밀도 |
| **Weight Decay** | 0 | 가중치 감쇠 |
| **Arm/Gripper Loss Ratio** | 0.01 | 팔/그리퍼 Loss 비율 |

**출처**: `RoboVLMs/configs/calvin_finetune/`, `RoboVLMs/main.py:136-309`

