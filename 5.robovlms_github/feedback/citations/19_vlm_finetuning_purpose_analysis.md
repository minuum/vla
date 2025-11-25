# 19. VLM Fine-tuning 목적 분석

## Critical Issue: LRN 토큰이 이미 학습된 VLM에서 나온다면, 무엇을 Fine-tuning하는가?

### 문제점
LRN (Learnable) 토큰이 이미 학습된 VLM에서 나온다면, 정확히 무엇을 fine-tuning하는지 불명확.

---

## 1. 핵심 발견사항

### **1.1 LRN 토큰의 정체**

```python
# RoboVLMs/robovlms/model/backbone/base_backbone.py:125-126
if self.action_space == "continuous":
    self.action_token = nn.Parameter(torch.zeros(self.hidden_size))
    self.action_token.requires_grad_(True)
```

**핵심**: LRN 토큰은 **새로 추가된 learnable parameter**입니다!

### **1.2 LRN 토큰의 역할**

```python
# RoboVLMs/robovlms/model/backbone/base_backbone.py:1102-1107
action_tokens = repeat(
    self.action_token,
    "d -> b n d",
    b=multimodal_embeds.shape[0],
    n=self.latent_num,
)
```

**핵심**: LRN 토큰은 **VLM의 입력으로 삽입**되어 VLM 내부에서 처리됩니다.

---

## 2. VLM Fine-tuning의 실제 목적

### **2.1 VLM이 학습해야 하는 것**

1. **LRN 토큰의 의미 학습**
   - LRN 토큰이 **어떤 액션을 의미하는지** 학습
   - Vision + Language + LRN 토큰의 **multimodal fusion** 학습

2. **Multimodal Context 이해**
   - 이미지 + 텍스트 + LRN 토큰의 **조합된 의미** 이해
   - 로봇 제어 상황에 맞는 **contextual representation** 학습

3. **Action-Context Mapping**
   - 특정 상황(이미지+텍스트)에서 **LRN 토큰이 어떤 액션을 나타내는지** 학습
   - **상황별 액션 매핑** 학습

### **2.2 구체적 학습 과정**

```python
# Training 시
Input: [Image Tokens] + [Text Tokens] + [LRN Tokens]
       ↓ VLM (Fine-tuning)
Output: [Fused Multimodal Representation]
        ↓ Policy Head
Action: [Predicted Actions]

# Loss 계산
Loss = MSE(Predicted Actions, Ground Truth Actions)
```

**핵심**: VLM이 **LRN 토큰의 의미를 학습**하여 적절한 액션을 예측할 수 있도록 fine-tuning

---

## 3. 실제 Fine-tuning되는 부분

### **3.1 Trainable Parameters**

```python
# RoboVLMs/robovlms/model/backbone/base_backbone.py:470-540
def _trainable_params_setup(self):
    model = self.model
    
    # 1. Backbone VLM 동결/해제
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
    
    # 2. Vision Tower 동결/해제
    if self.train_setup_configs.get("train_vision", False):
        self.vision_tower.requires_grad_(True)
    else:
        self.vision_tower.requires_grad_(False)
    
    # 3. LoRA 설정
    if self.train_setup_configs["lora_enable"]:
        # LoRA 파라미터만 학습 가능하도록 설정
        pass
    
    # 4. Text Embedding 동결/해제
    if self.train_setup_configs.get("train_text_embedding", False):
        self.word_embedding.requires_grad_(True)
    else:
        self.word_embedding.requires_grad_(False)
    
    # 5. Vision Resampler 동결/해제
    if self.use_vision_resampler:
        if not self.train_setup_configs.get("freeze_resampler", False):
            self.vision_resampler.requires_grad_(True)
        else:
            self.vision_resampler.requires_grad_(False)
    
    # 6. Action Head는 항상 학습
    if self.act_head is not None:
        self.act_head.requires_grad_(True)
```

### **3.2 실제 학습되는 부분들**

1. **LRN 토큰 자체** (`self.action_token`)
   - 새로운 learnable parameter
   - **항상 학습됨** (`requires_grad_(True)`)

2. **VLM의 일부 레이어**
   - `freeze_backbone=False`: 전체 VLM 학습
   - `train_decoder_layers=N`: 마지막 N개 레이어만 학습
   - `train_vision=True`: Vision encoder 학습
   - `train_text_embedding=True`: Text embedding 학습

3. **Vision Resampler** (선택적)
   - `freeze_resampler=False`: Vision resampler 학습

4. **LoRA Adapters** (선택적)
   - `lora_enable=True`: LoRA 파라미터만 학습

5. **Policy Head** (LSTM, FC, GPT 등)
   - **항상 학습됨**

---

## 4. Fine-tuning의 구체적 목적

### **4.1 LRN 토큰의 의미 학습**

```python
# 초기 상태
self.action_token = nn.Parameter(torch.zeros(self.hidden_size))
# LRN 토큰은 0으로 초기화됨 (의미 없음)

# Fine-tuning 후
# LRN 토큰이 특정 액션을 나타내는 의미 있는 벡터로 학습됨
```

### **4.2 Multimodal Fusion 학습**

```python
# VLM 내부에서
# [Image Features] + [Text Features] + [LRN Token] → [Fused Features]
# 
# Fine-tuning을 통해 VLM이 학습하는 것:
# 1. LRN 토큰이 어떤 액션을 의미하는지
# 2. 이미지+텍스트+액션의 조합된 의미
# 3. 상황별 적절한 액션 예측
```

### **4.3 Context-Action Mapping 학습**

```python
# 예시: "물체를 잡아라" + [이미지] + [LRN] → [잡기 액션]
# 
# Fine-tuning을 통해 VLM이 학습하는 것:
# - 특정 상황에서 LRN 토큰이 어떤 액션을 나타내는지
# - 이미지와 텍스트의 조합에 따른 액션 매핑
# - 로봇 제어 상황에 맞는 이해
```

---

## 5. 왜 VLM Fine-tuning이 필요한가?

### **5.1 기존 VLM의 한계**

- **일반적인 VLM**: 이미지 + 텍스트 → 텍스트 생성
- **로봇 제어 VLM**: 이미지 + 텍스트 + 액션 → 액션 예측
- **새로운 도메인**: 로봇 제어는 VLM이 처음 접하는 영역

### **5.2 Fine-tuning의 필요성**

1. **LRN 토큰 이해**: VLM이 LRN 토큰의 의미를 학습
2. **Multimodal Fusion**: 이미지+텍스트+액션의 조합 이해
3. **Domain Adaptation**: 로봇 제어 도메인에 맞는 이해
4. **Context-Action Mapping**: 상황별 액션 매핑 학습

### **5.3 Fine-tuning vs. 새로 학습**

- **Fine-tuning**: 기존 VLM의 지식을 활용하면서 로봇 제어에 특화
- **새로 학습**: 처음부터 로봇 제어에 맞게 학습 (비효율적)

---

## 6. 구체적 학습 과정

### **6.1 Forward Pass**

```python
# 1. 입력 준비
image_features = vision_encoder(images)  # [batch, seq_len, hidden_size]
text_features = text_encoder(text)       # [batch, seq_len, hidden_size]
lrn_tokens = repeat(self.action_token, "d -> b n d", b=batch, n=latent_num)

# 2. VLM에 입력
multimodal_input = [image_features, text_features, lrn_tokens]
vlm_output = vlm_model(multimodal_input)  # [batch, seq_len, hidden_size]

# 3. LRN 토큰 위치 추출
lrn_features = vlm_output[lrn_token_mask]  # [batch, latent_num, hidden_size]

# 4. Policy Head로 액션 예측
predicted_actions = policy_head(lrn_features)  # [batch, latent_num, action_dim]
```

### **6.2 Loss 계산**

```python
# Ground Truth와 비교
loss = MSE(predicted_actions, ground_truth_actions)

# Backward Pass
loss.backward()

# Parameter 업데이트
# - self.action_token (LRN 토큰)
# - VLM의 일부 레이어 (설정에 따라)
# - Policy Head
```

### **6.3 학습 결과**

```python
# Fine-tuning 전
self.action_token = torch.zeros(hidden_size)  # 의미 없음

# Fine-tuning 후  
self.action_token = learned_vector  # 특정 액션을 나타내는 의미 있는 벡터
```

---

## 7. 핵심 결론

### **7.1 VLM Fine-tuning의 목적**

1. **LRN 토큰의 의미 학습**: 새로운 learnable parameter 학습
2. **Multimodal Fusion 학습**: 이미지+텍스트+액션 조합 이해
3. **Domain Adaptation**: 로봇 제어 도메인에 특화
4. **Context-Action Mapping**: 상황별 액션 매핑 학습

### **7.2 학습되는 부분**

- **LRN 토큰**: 항상 학습 (`self.action_token`)
- **VLM 레이어**: 설정에 따라 전체/일부 학습
- **Policy Head**: 항상 학습
- **Vision Resampler**: 선택적 학습
- **LoRA Adapters**: 선택적 학습

### **7.3 왜 Fine-tuning이 필요한가?**

- **기존 VLM**: 이미지+텍스트 → 텍스트
- **로봇 VLM**: 이미지+텍스트+액션 → 액션
- **새로운 도메인**: 로봇 제어는 VLM이 처음 접하는 영역
- **효율성**: 기존 지식 활용하면서 도메인 특화

**출처**:
- `RoboVLMs/robovlms/model/backbone/base_backbone.py:125-126, 470-540`
- `RoboVLMs/robovlms/model/backbone/base_backbone.py:1102-1107`
- `RoboVLMs/robovlms/model/backbone/roboflamingo.py:131-156`

**핵심**: LRN 토큰은 **새로 추가된 learnable parameter**이며, VLM fine-tuning의 목적은 **LRN 토큰의 의미를 학습**하고 **multimodal fusion**을 통해 **로봇 제어에 특화**된 이해를 얻는 것입니다.
