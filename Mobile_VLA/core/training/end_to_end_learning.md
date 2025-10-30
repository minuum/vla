# End-to-End 학습 과정

## 1. 학습 파이프라인 전체 흐름

### 1.1 전체 학습 과정

```
1. 데이터 로드
   ↓
2. Image → Vision Tokens (VLM Vision Encoder)
   ↓
3. Text → Text Tokens (VLM Tokenizer)
   ↓
4. [LRN] Token 추가
   ↓
5. Multi-modal Fusion (VLM Backbone)
   ↓
6. [LRN] Token 추출
   ↓
7. LSTM에 [LRN] 입력
   ↓
8. 7-DOF Action 예측
   ↓
9. Loss 계산 (MSE + BCE)
   ↓
10. Backpropagation (VLM + LSTM 동시 업데이트)
```

### 1.2 핵심 학습 원리

**End-to-End 학습**:
- VLM과 LSTM이 동시에 학습
- Gradient가 전체 파이프라인을 통해 전파
- Multi-modal 정보가 자연스럽게 융합

## 2. Forward Pass 상세

### 2.1 forward_continuous() 핵심 로직

```python
def forward_continuous(
    self,
    vision_x: torch.Tensor,           # [batch, seq_len, 2, 3, 224, 224]
    lang_x: torch.Tensor,             # [batch, seq_len, text_len]
    action_labels: torch.Tensor,      # [batch, seq_len, 7]
    attention_mask: torch.Tensor,     # [batch, seq_len, text_len]
    **kwargs
):
    loss = {}
    bs, seq_len = vision_x.shape[:2]
    action_space = self.act_head_configs.get("action_space", "continuous")
    
    # 1. 히스토리 타입 설정
    history_type = self.act_head_configs.get("history_type", "post")
    
    if history_type in ["post", "pre"]:
        # 시퀀스 차원 재구성
        vision_x = vision_x.reshape(bs * seq_len, *vision_x.shape[2:]).unsqueeze(1)
        lang_x = lang_x.unsqueeze(1).repeat(1, seq_len, 1).flatten(0, 1)
        attention_mask = attention_mask.unsqueeze(1).repeat(1, seq_len, 1).flatten(0, 1)
    
    # 2. Text Embedding 생성
    input_embeds = self.word_embedding(lang_x)  # [batch*seq_len, text_len, hidden_size]
    
    # 3. Multi-modal Input 생성
    multimodal_embeds, mutlimodal_labels, multimodal_attention_mask, _ = self.merge_multi_modal_input(
        input_embeds,
        vision_x,
        labels=None,
        attention_mask=attention_mask,
        insert_idx=bos_offset,
    )
    
    # 4. Action Token 추가 (연속 액션의 경우)
    if action_space == "continuous":
        insert_idx = multimodal_embeds.shape[1] - int(self.tokenizer.eos_token is not None)
        
        action_tokens = repeat(
            self.action_token,  # [hidden_size]
            "d -> b n d",
            b=multimodal_embeds.shape[0],
            n=self.latent_num,  # 보통 1
        )
        
        multimodal_embeds, mutlimodal_labels, multimodal_attention_mask, action_token_mask = self.merge_multi_modal_input(
            multimodal_embeds,
            action_tokens,
            mutlimodal_labels,
            multimodal_attention_mask,
            is_image=False,
            insert_idx=insert_idx,
            fill_zero=self.act_head_configs.get("fill_zero", False),
        )
    
    # 5. VLM Backbone 통과
    output = self.model(
        input_ids=None,
        attention_mask=multimodal_attention_mask,
        position_ids=position_ids,
        inputs_embeds=multimodal_embeds,
        use_cache=use_cache,
        output_hidden_states=True,
    )
    
    # 6. Hidden States 추출
    output_hs = output.hidden_states[-1].clone()  # [batch*seq_len, total_len, hidden_size]
    
    # 7. Action Token 추출
    if action_space == "continuous":
        action_hs = output_hs[action_token_mask].reshape(
            bs, seq_len, self.latent_num, -1
        )  # [batch, seq_len, 1, hidden_size]
    
    # 8. Policy Head (LSTM) Forward
    action_logits, action_loss = self._forward_action_head(
        action_hs, action_labels, action_mask
    )
    
    # 9. Loss 업데이트
    if mode == "train":
        self._update_loss(loss, action_loss, "act")
        loss = self._format_loss(loss)
    else:
        return action_logits
    
    return loss
```

### 2.2 Multi-modal Input 생성

```python
def merge_multi_modal_input(
    self,
    input_embeds: torch.Tensor,
    vision_x: torch.Tensor,
    labels: torch.Tensor = None,
    attention_mask: torch.Tensor = None,
    insert_idx: int = 0,
    **kwargs
):
    """
    Text, Vision, Action 토큰을 하나의 시퀀스로 결합
    """
    # 1. Vision Encoding
    vision_embeds = self.encode_images(vision_x)  # [batch, num_patches, hidden_size]
    
    # 2. 토큰 순서: [Text Tokens] + [Vision Tokens] + [Action Tokens]
    multimodal_embeds = torch.cat([
        input_embeds[:, :insert_idx],      # Text tokens (BOS 이후)
        vision_embeds,                     # Vision tokens
        input_embeds[:, insert_idx:],      # Text tokens (나머지)
    ], dim=1)
    
    # 3. Attention Mask 업데이트
    if attention_mask is not None:
        vision_mask = torch.ones(
            vision_embeds.shape[:2], 
            dtype=attention_mask.dtype, 
            device=attention_mask.device
        )
        multimodal_attention_mask = torch.cat([
            attention_mask[:, :insert_idx],
            vision_mask,
            attention_mask[:, insert_idx:],
        ], dim=1)
    else:
        multimodal_attention_mask = None
    
    return multimodal_embeds, labels, multimodal_attention_mask, None
```

## 3. Loss Function과 Gradient Flow

### 3.1 Loss 계산

```python
def _forward_action_head(
    self,
    action_tokens: torch.Tensor,      # [batch, seq_len, 1, hidden_size]
    action_labels: torch.Tensor,      # [batch, seq_len, 7]
    action_mask: torch.Tensor = None,
    **kwargs,
):
    # 1. LSTM Policy Head Forward
    action = self.act_head(
        action_tokens, 
        actions=action_labels, 
        action_masks=action_mask, 
        **kwargs
    )  # [batch, seq_len, 7]
    
    action_loss = None
    if action_labels is not None:
        # 2. Label 정규화 및 마스킹
        action, action_labels, action_mask = self.act_head.get_labels(
            action, action_labels, action_mask, tok_seq=action_tokens, **kwargs
        )
        
        # 3. Loss 계산
        action_loss = self.act_head.loss(action, action_labels, action_mask)
    
    return action, action_loss
```

### 3.2 Continuous Action Loss (논문 Equation 7)

```python
# LSTM Policy Head Loss 계산
def loss(self, predicted_action, ground_truth, action_mask):
    """
    Continuous Action Loss (논문 Equation 7)
    """
    # 1. Pose Loss (MSE) - 처음 6차원 (translation + rotation)
    loss_pose = F.mse_loss(
        predicted_action[:, :6], 
        ground_truth[:, :6]
    )
    
    # 2. Gripper Loss (BCE) - 마지막 1차원 (gripper)
    loss_gripper = F.binary_cross_entropy_with_logits(
        predicted_action[:, 6], 
        ground_truth[:, 6]
    )
    
    # 3. 가중합
    total_loss = loss_pose + lambda_gripper * loss_gripper  # lambda_gripper = 0.01
    
    return total_loss
```

**왜 MSE + BCE인가?**
- **MSE (처음 6차원)**: Translation + Rotation은 연속 값 (회귀 문제)
- **BCE (마지막 1차원)**: Gripper는 binary (열기 -1 / 닫기 1)
- **λ = 0.01**: Gripper loss의 가중치를 낮춰 pose 학습 우선

### 3.3 Gradient Flow: VLM과 LSTM 동시 학습

```
Loss (MSE + BCE)
  ↓ (Backpropagation)
Policy Head (LSTM)
  ↓
[LRN] Token Output
  ↓
VLM Backbone (Attention Layers)
  ↓
[LRN] Token Embedding (학습 가능)
  ↓
Vision Encoder (학습 가능)
  ↓
Text Encoder (학습 가능)
```

**핵심**:
- **End-to-End 학습**: Loss에서 VLM의 Vision Encoder까지 gradient 전파
- **Action Token 최적화**: [LRN]의 embedding이 "어떤 정보를 VLM에서 추출해야 하는지" 학습
- **VLM 파라미터 업데이트**: Vision/Text Encoder가 로봇 태스크에 맞춰 재학습

## 4. 학습 가능한 파라미터 설정

### 4.1 Trainable Parameters Setup

```python
def _trainable_params_setup(self):
    model = self.model  # VLM Backbone
    
    # 1. 백본 모델 설정
    if self.train_setup_configs["freeze_backbone"]:
        model.requires_grad_(False)  # 동결
    else:
        model.requires_grad_(True)   # ✅ 전체 학습
    
    # 2. Vision Encoder 설정
    if self.train_setup_configs.get("train_vision", False):
        self.vision_tower.requires_grad_(True)  # ✅ 학습
    else:
        self.vision_tower.requires_grad_(False)
    
    # 3. LoRA 설정 (Full-FT는 skip)
    if self.train_setup_configs["lora_enable"]:
        # LoRA 적용 (RoboVLMs는 사용 안 함)
        pass
    
    # 4. Action Token 학습 (항상 True)
    self.action_token.requires_grad_(True)  # ✅ 학습
    
    # 5. Policy Head 학습 (항상 True)
    self.act_head.requires_grad_(True)  # ✅ 학습
```

### 4.2 Full Fine-tuning 설정

```json
{
    "train_setup": {
        "predict_action": true,
        "freeze_backbone": false,      // VLM 전체 학습
        "train_vision": true,          // Vision Encoder 학습
        "train_text_embedding": true,  // Text Embedding 학습
        "lora_enable": false,          // Full-FT
        "learning_rate": 2e-5,
        "weight_decay": 0
    }
}
```

**결론**:
- **VLM Vision Encoder**: ✅ 학습
- **VLM Text Encoder**: ✅ 학습
- **VLM Backbone (Attention)**: ✅ 학습
- **[LRN] Token**: ✅ 학습
- **LSTM Policy Head**: ✅ 학습

**모든 것이 동시에 학습됩니다!**

## 5. 학습 설정과 하이퍼파라미터

### 5.1 기본 학습 설정

```python
# 기본 학습 설정
{
    "batch_size": 8,
    "learning_rate": 2e-5,
    "weight_decay": 0,
    "num_epochs": 10,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 1000,
    "max_grad_norm": 1.0
}
```

### 5.2 Mixed Precision Training

```python
# Mixed Precision Training 설정
{
    "mixed_precision": true,
    "fp16": true,
    "bf16": false,
    "gradient_checkpointing": true
}
```

### 5.3 Learning Rate Scheduler

```python
# Learning Rate Scheduler
{
    "scheduler": "cosine",
    "warmup_steps": 1000,
    "num_training_steps": 50000,
    "min_lr": 1e-6
}
```

## 6. 메모리 효율성 최적화

### 6.1 Gradient Checkpointing

```python
# 메모리 효율성을 위한 Gradient Checkpointing
if self.gradient_checkpointing:
    self.model.gradient_checkpointing_enable()
```

### 6.2 Batch Size 조정

```python
# 메모리에 따른 Batch Size 조정
{
    "batch_size": 8,                    # 기본값
    "gradient_accumulation_steps": 4,   # 메모리 부족시 사용
    "effective_batch_size": 32          # 8 * 4 = 32
}
```

### 6.3 모델 병렬화

```python
# 모델 병렬화 설정
{
    "model_parallel": true,
    "tensor_parallel_size": 2,
    "pipeline_parallel_size": 1
}
```

## 7. 학습 모니터링

### 7.1 Loss 추적

```python
# Loss 추적
def _format_loss(self, loss):
    """Loss 포맷팅 및 총합 계산"""
    _loss = 0
    _keys = list(loss.keys())
    
    for k in _keys:
        if "loss" in k:
            _loss += loss[k]
    
    loss["loss"] = _loss
    return loss
```

### 7.2 메트릭 모니터링

```python
# 학습 메트릭
{
    "train_loss": "MSE + BCE",
    "val_loss": "MSE + BCE",
    "action_accuracy": "Gripper 정확도",
    "pose_error": "Translation + Rotation 오차"
}
```

## 8. 학습 과정 요약

### 8.1 핵심 학습 원리

1. **Multi-modal Fusion**: VLM이 이미지와 텍스트를 융합하여 [LRN] 토큰에 액션 정보 인코딩
2. **End-to-End 학습**: Loss에서 VLM까지 gradient 전파
3. **동시 최적화**: VLM과 LSTM이 동시에 로봇 태스크에 맞춰 학습

### 8.2 학습 결과

- **Vision Encoder**: 로봇 태스크에 유용한 이미지 특징 학습
- **Text Encoder**: 명령과 액션의 관계 학습
- **[LRN] Token**: Multi-modal 정보 융합 학습
- **LSTM**: 7-DOF 액션 예측 + 히스토리 모델링

## 9. 참고 자료

- `RoboVLMs/robovlms/model/backbone/base_backbone.py`: 기본 VLM + LSTM 통합
- `RoboVLMs/robovlms/model/policy_head/`: Policy Head 구현
- `RoboVLMs/robovlms/train/`: 학습 스크립트
- `RoboVLMs/configs/calvin_finetune/`: CALVIN 학습 설정
