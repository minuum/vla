# Critical Issue #3: [LRN] Token의 정확한 위치와 처리

## 문제점

"[LRN] 토큰이 마지막에 추가된다"고 했지만, **정확히 어디에**, **몇 개**, **어떻게 처리되는지** 불명확.

## 정확한 사실

### 1. [LRN] Token 생성 (단일 파라미터)

```python
# BaseRoboVLM.__init__()
if self.action_space == "continuous":
    # 단일 벡터 (shape: [hidden_size])
    self.action_token = nn.Parameter(torch.zeros(self.hidden_size))
    self.action_token.requires_grad_(True)
```

**출처**: `RoboVLMs/robovlms/model/backbone/base_backbone.py:124-126`

### 2. Batch별로 복제

```python
# forward_continuous()
action_tokens = repeat(
    self.action_token,  # [hidden_size]
    "d -> b n d",
    b=batch_size,       # 배치 크기만큼 복제
    n=self.latent_num   # latent 개수 (기본 1)
)
# 결과 shape: [batch_size, 1, hidden_size]
```

**출처**: `RoboVLMs/robovlms/model/backbone/base_backbone.py:1102-1107`

### 3. 삽입 위치 (Encoder-Decoder 구조)

```python
# merge_multi_modal_input() - PaliGemma, Kosmos 등
insert_idx = multimodal_embeds.shape[1] - int(self.tokenizer.eos_token is not None)
# EOS 토큰이 있으면 그 앞에 삽입, 없으면 맨 끝에 삽입

# 예: [BOS, Text..., IMG..., [LRN0], EOS0] t0
#                              ↑ 여기!
```

### 4. 추출 위치

```python
# VLM 출력에서 [LRN] 추출
vlm_output = self.model(inputs_embeds=multimodal_embeds, ...)
# shape: [batch, total_seq_len, hidden_size]

# 마지막 토큰이 [LRN]의 출력
action_token_output = vlm_output[:, -1, :]
# shape: [batch, hidden_size]
```

## 핵심 확인사항

- **[LRN]은 1개의 학습 가능한 파라미터**
- 배치마다 복제되어 사용
- VLM 통과 후 **마지막 위치의 출력**을 Policy Head에 입력

## 상세 처리 과정

### 1. [LRN] Token 초기화

```python
# BaseRoboVLM.__init__()
if self.action_space == "continuous":
    # 학습 가능한 파라미터로 초기화
    self.action_token = nn.Parameter(torch.zeros(self.hidden_size))
    self.action_token.requires_grad_(True)
    
    # latent_num 설정 (보통 1)
    self.latent_num = act_head_configs.get("latent", 1)
```

### 2. Multi-modal Input 생성

```python
# forward_continuous()
def merge_multi_modal_input(self, input_embeds, vision_x, ...):
    # 1. Text Embedding
    text_embeds = self.word_embedding(lang_x)
    
    # 2. Vision Encoding
    vision_embeds = self.encode_images(vision_x)
    
    # 3. [LRN] Token 추가
    action_tokens = repeat(
        self.action_token,  # [hidden_size]
        "d -> b n d",
        b=multimodal_embeds.shape[0],
        n=self.latent_num  # 보통 1
    )
    
    # 4. 삽입 위치 계산
    insert_idx = multimodal_embeds.shape[1] - int(self.tokenizer.eos_token is not None)
    
    # 5. Multi-modal Input 생성
    multimodal_embeds = torch.cat([
        multimodal_embeds[:, :insert_idx],  # Text tokens (BOS 이후)
        action_tokens,                      # [LRN] tokens
        multimodal_embeds[:, insert_idx:],  # Text tokens (나머지)
    ], dim=1)
    
    return multimodal_embeds, labels, attention_mask, action_token_mask
```

### 3. VLM 통과 및 추출

```python
# VLM Backbone 통과
output = self.model(
    input_ids=None,
    attention_mask=multimodal_attention_mask,
    inputs_embeds=multimodal_embeds,
    output_hidden_states=True,
)

# Hidden States 추출
output_hs = output.hidden_states[-1].clone()
# shape: [batch, total_seq_len, hidden_size]

# [LRN] Token 추출 (마지막 위치)
action_hs = output_hs[action_token_mask].reshape(
    bs, seq_len, self.latent_num, -1
)
# shape: [batch, seq_len, 1, hidden_size]
```

### 4. Policy Head 입력

```python
# Policy Head (LSTM) Forward
action_logits, action_loss = self._forward_action_head(
    action_hs, action_labels, action_mask
)

# LSTM이 [LRN] 토큰을 받아 액션 예측
predicted_action = self.act_head(action_hs)
# shape: [batch, seq_len, 7]
```

## 토큰 시퀀스 예시

### 입력 시퀀스

```
[BOS] "pick" "up" "the" "red" "block" [IMG_1] [IMG_2] ... [IMG_N] [LRN] [EOS]
  ↑     ↑     ↑     ↑     ↑      ↑      ↑      ↑        ↑      ↑     ↑
  0     1     2     3     4      5      6      7        N     N+1   N+2
```

### VLM 처리 후

```
[BOS] "pick" "up" "the" "red" "block" [IMG_1] [IMG_2] ... [IMG_N] [LRN] [EOS]
  ↑     ↑     ↑     ↑     ↑      ↑      ↑      ↑        ↑      ↑     ↑
  0     1     2     3     4      5      6      7        N     N+1   N+2
                                                         ↑
                                                    Policy Head 입력
```

## 정리

### [LRN] Token의 특징
1. **단일 파라미터**: `self.action_token` (shape: [hidden_size])
2. **배치 복제**: 각 배치마다 동일한 토큰 사용
3. **삽입 위치**: EOS 토큰 앞 (또는 맨 끝)
4. **추출 위치**: VLM 출력의 마지막 토큰
5. **학습 가능**: `requires_grad_(True)`

### 처리 과정
1. **초기화**: 0으로 초기화된 학습 가능한 파라미터
2. **복제**: 배치 크기만큼 복제
3. **삽입**: Multi-modal input에 삽입
4. **VLM 통과**: Self-attention을 통해 정보 융합
5. **추출**: 마지막 위치에서 추출
6. **Policy Head**: LSTM에 입력하여 액션 예측

### 핵심 포인트
- **[LRN]은 1개**: 배치마다 복제되지만 원본은 1개
- **마지막 위치**: VLM 출력에서 마지막 토큰이 [LRN]의 결과
- **학습 가능**: Fine-tuning 과정에서 최적화됨
