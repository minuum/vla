# Critical Issue #5: Action Chunk의 정확한 처리

## 문제점

"Action chunk를 예측한다"고 했지만, **정확히 무엇을 의미하는지**, **어떻게 생성되는지**, **왜 매 시간 단계마다 N개를 예측하는지** 불명확.

## 정확한 사실

### 1. Action Chunk 생성 (Ground Truth)

```python
# Config
{
    "fwd_pred_next_n": 10  # 미래 10개 액션 예측
}

# 데이터 로더에서 Sliding Window로 생성
action_chunk = action_tensors.unfold(1, self.fwd_pred_next_n, 1).permute(0, 1, 3, 2)
# Input shape:  [batch, window_size, action_dim]
# Output shape: [batch, window_size - fwd_pred_next_n + 1, fwd_pred_next_n, action_dim]
```

**출처**: `RoboVLMs/robovlms/data/calvin_dataset.py:884-887`

### 2. Sliding Window 방식 (Ground Truth 생성)

```
시간 단계:  t0  t1  t2  t3  t4  t5  t6  t7  t8  t9  t10 t11
Action:     a0  a1  a2  a3  a4  a5  a6  a7  a8  a9  a10 a11

Ground Truth Chunks:
  t0 시점 관측 → Label: [a0, a1, a2, ..., a9]   (10개)
  t1 시점 관측 → Label: [a1, a2, a3, ..., a10]  (10개)
  t2 시점 관측 → Label: [a2, a3, a4, ..., a11]  (10개)
```

**핵심**:
- **각 시간 단계에서 "현재부터 미래 N개" 액션을 Ground Truth로 설정**
- t0에서는 a0~a9, t1에서는 a1~a10 (window가 1칸씩 슬라이드)

### 3. LSTM 예측 (모든 시간 단계에서 N개 예측)

```python
# LSTMDecoder
class LSTMDecoder(BasePolicyHead):
    def forward(self, tok_seq):
        # tok_seq: [batch, window_size, hidden_size]
        # 예: window_size=8이면 8개 시간 단계의 [LRN] 토큰
        
        # LSTM 통과
        output, (h_n, c_n) = self.lstm(tok_seq)
        # output: [batch, window_size, lstm_hidden]
        
        # 각 시간 단계마다 독립적인 출력 생성
        # 각 시간 단계에서 fwd_pred_next_n개 액션 예측
        predicted_actions = self.fc(output)
        # shape: [batch, window_size, fwd_pred_next_n * action_dim]
        
        # Reshape
        predicted_actions = predicted_actions.view(
            batch, window_size, fwd_pred_next_n, action_dim
        )
        # 최종 shape: [batch, window_size, fwd_pred_next_n, action_dim]
        
        return predicted_actions
```

**출처**: `RoboVLMs/robovlms/model/policy_head/lstm_decoder.py` (구조 기반)

### 4. 구체적 예시: window_size=8, fwd_pred_next_n=10

```python
# Training 시점
Input:
  [LRN]_t0, [LRN]_t1, [LRN]_t2, ..., [LRN]_t7  (8개 시간 단계)
  ↓ LSTM
Output (Predicted Actions):
  t0 예측: [pred_a0, pred_a1, ..., pred_a9]    (10개) ← t0 시점에서 미래 10개
  t1 예측: [pred_a1, pred_a2, ..., pred_a10]   (10개) ← t1 시점에서 미래 10개
  t2 예측: [pred_a2, pred_a3, ..., pred_a11]   (10개) ← t2 시점에서 미래 10개
  ...
  t7 예측: [pred_a7, pred_a8, ..., pred_a16]   (10개) ← t7 시점에서 미래 10개

Ground Truth (Label):
  t0 label: [a0, a1, ..., a9]
  t1 label: [a1, a2, ..., a10]
  t2 label: [a2, a3, ..., a11]
  ...
  t7 label: [a7, a8, ..., a16]

Loss 계산:
  Loss = MSE(t0_pred, t0_label) + MSE(t1_pred, t1_label) + ... + MSE(t7_pred, t7_label)
```

### 5. 왜 매 시간 단계마다 N개를 예측하는가?

**이유 1: Teacher Forcing 효과**
```
t0에서 예측 실패해도 → t1은 ground truth [LRN]_t1으로 예측
→ 에러 누적 방지
```

**이유 2: Temporal Consistency 학습**
```
t0 예측: [a0, a1, a2, ..., a9]
t1 예측: [a1, a2, ..., a10]
t8 예측: [a8, a9, ..., a18]
→ a1~a9가 겹침 (overlap)
→ 일관성 있는 액션 시퀀스 학습
```

**이유 3: 데이터 효율성**
```
window_size=8 → 8개 시간 단계에서 각각 10개 예측
→ 단일 forward pass로 80개 액션 예측 학습
→ 학습 효율 대폭 증가
```

**이유 4: MPC(Model Predictive Control) 구조**
```
실제 로봇 제어:
  t0 시점: 10개 예측 → 첫 번째 액션만 실행
  t1 시점: 새로 10개 예측 → 첫 번째 액션만 실행
→ Receding Horizon 방식
```

### 6. Inference 시 동작 (중요!)

```python
# Inference: 현재 시점에서만 예측
Input:
  [LRN]_current  (1개)
  ↓ LSTM (이전 hidden state 유지)
Output:
  predicted_actions: [a_current, a_current+1, ..., a_current+9]  (10개)

실제 실행:
  predicted_actions[0]만 실행  ← 첫 번째 액션만 사용

다음 단계:
  새로운 관측 → 새로운 [LRN] → 다시 10개 예측 → 첫 번째만 실행
```

**핵심**:
- **Training**: 모든 시간 단계에서 N개 예측 (병렬 학습)
- **Inference**: 매 단계마다 N개 예측하지만 **첫 번째만 실행** (MPC 방식)

### 7. 정리: Action Chunk의 정확한 의미

| **구분** | **내용** |
|----------|----------|
| **Ground Truth** | Sliding window로 각 시간 단계마다 "현재~미래 N개" 추출 |
| **모델 예측** | LSTM이 각 시간 단계마다 N개 액션 동시 예측 |
| **Loss 계산** | 각 시간 단계의 N개 예측과 N개 label 간 MSE |
| **Training 효과** | 여러 시간 단계에서 겹치는 액션 예측 → 일관성 학습 |
| **Inference 사용** | N개 예측하지만 첫 번째만 실행 (MPC) |

**왜 이렇게 하는가?**

1. **에러 누적 방지**: Teacher forcing으로 각 단계 독립적 학습
2. **일관성 학습**: Overlap되는 액션으로 temporal consistency 확보
3. **데이터 효율**: 단일 forward로 많은 예측 생성
4. **실제 제어**: MPC처럼 매 단계 재계획

## 실제 코드 근거

### Ground Truth Action Chunk 생성 코드

```python
# RoboVLMs/robovlms/data/calvin_dataset.py:199-200
self.act_step = fwd_pred_next_n + 1
self.fwd_pred_next_n = fwd_pred_next_n

# RoboVLMs/robovlms/data/calvin_dataset.py:884-887
action_chunck = action_tensors.unfold(1, self.fwd_pred_next_n, 1).permute(0, 1, 3, 2)
# unfold(dim=1, size=fwd_pred_next_n, step=1)
# - dim=1: 시간 차원(window_size)에서 sliding
# - size: 각 window에서 추출할 개수 (fwd_pred_next_n)
# - step: 1칸씩 슬라이드
# 결과: [batch, window_size-fwd_pred_next_n+1, action_dim, fwd_pred_next_n]
# permute(0,1,3,2): [batch, window_size-fwd_pred_next_n+1, fwd_pred_next_n, action_dim]
action_mask = action_mask.unfold(1, self.fwd_pred_next_n, 1)
```

### LSTM Decoder 초기화

```python
# RoboVLMs/robovlms/model/policy_head/base_policy.py:387-426
class LSTMDecoder(BasePolicyHead):
    def __init__(
        self,
        in_features,
        action_dim,
        down_sample,
        latent,
        fwd_pred_next_n,        # 미래 N개 액션 예측
        window_size,            # LSTM 입력 시퀀스 길이
        hidden_size=1024,
        num_layers=4,
        policy_rnn_dropout_p=0.0,
        **kwargs,
    ):
        super().__init__(in_features, action_dim, **kwargs)
        self.down_sample = down_sample
        self.latent = latent
        self.window_size = window_size
        self.history_len = window_size
        self.fwd_pred_next_n = fwd_pred_next_n  # 저장
        self.history_memory = []
        self.hidden_size = hidden_size
        
        # LSTM 정의
        self.rnn = lstm_decoder(
            in_features * latent,
            hidden_size * latent,
            num_layers,
            policy_rnn_dropout_p
        )
        
        # Action Head: fwd_pred_next_n * (action_dim - 1) 출력
        self.actions = MLPTanhHead(
            self.hidden_size * latent,
            fwd_pred_next_n * (self.action_dim - 1)
        )
        
        # Gripper Head: fwd_pred_next_n 출력
        self.gripper = MLPSigmoidHead(
            self.hidden_size * latent,
            fwd_pred_next_n
        )
        self.hidden_state = None
```

### LSTM Forward Pass (핵심)

```python
# RoboVLMs/robovlms/model/policy_head/base_policy.py:432-484
def forward(self, tok_seq, h_0=None, **kwargs):
    # tok_seq: [batch, window_size, hidden_size]
    # VLM에서 추출한 [LRN] 토큰 시퀀스
    
    # Down-sampling (필요 시)
    if self.down_sample == "pooling":
        bs, seq_len = tok_seq.shape[:2]
        tok_seq = rearrange(tok_seq, "b l n d-> (b l) n d")
        tok_seq = self.global_1d_pool(tok_seq.permute(0, 2, 1))
        tok_seq = rearrange(tok_seq, "(b l) d n -> b l (n d)", b=bs, l=seq_len)
    elif self.down_sample == "none":
        tok_seq = rearrange(tok_seq, "b l n d-> b l (n d)")
    
    # LSTM 통과
    if h_0 is None:
        x, h_n = self.rnn(tok_seq)
        # x: [batch, window_size, hidden_size * latent]
        # h_n: hidden state
    else:
        x, h_n = self.rnn(tok_seq, h_0)
    
    if kwargs.get("is_train", True):
        self.hidden_state = h_n
    
    # Action 예측: 각 시간 단계에서 fwd_pred_next_n개 예측
    actions = self.actions(x)
    # actions: [batch, window_size, fwd_pred_next_n * (action_dim - 1)]
    gripper = self.gripper(x)
    # gripper: [batch, window_size, fwd_pred_next_n]
    
    # Reshape: [batch, window_size, fwd_pred_next_n, action_dim]
    actions = rearrange(
        actions,
        "b l (n d) -> b l n d",
        n=self.fwd_pred_next_n
    )
    gripper = rearrange(
        gripper,
        "b l (n d) -> b l n d",
        n=self.fwd_pred_next_n
    )
    
    return actions, gripper
```

## 정리

### Action Chunk의 핵심

1. **Ground Truth**: Sliding window로 각 시간 단계마다 "현재~미래 N개" 추출
2. **모델 예측**: LSTM이 각 시간 단계마다 N개 액션 동시 예측
3. **Loss 계산**: 각 시간 단계의 N개 예측과 N개 label 간 MSE
4. **Training 효과**: 여러 시간 단계에서 겹치는 액션 예측 → 일관성 학습
5. **Inference 사용**: N개 예측하지만 첫 번째만 실행 (MPC)

### 왜 이렇게 하는가?

1. **에러 누적 방지**: Teacher forcing으로 각 단계 독립적 학습
2. **일관성 학습**: Overlap되는 액션으로 temporal consistency 확보
3. **데이터 효율**: 단일 forward로 많은 예측 생성
4. **실제 제어**: MPC처럼 매 단계 재계획
