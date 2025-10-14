# RoboVLMs 학습/추론 파이프라인 분석 (한글)

## 코드 분석 개요

RoboVLMs의 `BaseRoboVLM` 클래스를 통해 핵심 학습/추론 파이프라인과 주요 학습 아이디어를 분석합니다. 이 코드는 VLM을 VLA로 변환하는 핵심 메커니즘을 보여줍니다.

## 핵심 학습 아이디어

### 1. 멀티모달 토큰 융합 (Multi-modal Token Fusion)

#### 핵심 개념
```python
def merge_multi_modal_input(self, input_embeds, multimodal_feats, labels, attention_mask, is_image=True, insert_idx=1):
    """
    비전, 언어, 액션 토큰을 하나의 시퀀스로 융합
    """
    if is_image:
        rgb_feats = self.encode_images(multimodal_feats)
        # 이미지 토큰에 시작/끝 토큰 추가
        if self.start_image_token_id is not None:
            image_start_embed = self.word_embedding(self.start_image_token_id)
            image_end_embed = self.word_embedding(self.end_image_token_id)
            rgb_feats = torch.cat([image_start_embed, rgb_feats, image_end_embed], dim=2)
    
    # 시퀀스 차원 평탄화
    rgb_feats = rearrange(rgb_feats, "b l n d -> b (l n) d")
    
    # 멀티모달 임베딩 결합
    multimodal_embeds = torch.cat([input_embeds[:, :insert_idx], rgb_feats, input_embeds[:, insert_idx:]], dim=1)
```

#### 학습 아이디어
- **통합 토큰 시퀀스**: 비전, 언어, 액션을 하나의 연속된 토큰 시퀀스로 처리
- **위치 기반 삽입**: 특정 위치에 모달리티별 토큰 삽입으로 구조적 정보 보존
- **어텐션 마스크 관리**: 각 모달리티의 어텐션 패턴을 명시적으로 제어

### 2. 액션 토큰 학습 (Action Token Learning)

#### 연속 액션 공간
```python
if action_space == "continuous":
    # 학습 가능한 액션 토큰 생성
    self.action_token = nn.Parameter(torch.zeros(self.hidden_size))
    self.action_token.requires_grad_(True)
    
    # 액션 토큰을 시퀀스에 삽입
    action_tokens = repeat(self.action_token, "d -> b n d", b=multimodal_embeds.shape[0], n=self.latent_num)
    multimodal_embeds = self.merge_multi_modal_input(multimodal_embeds, action_tokens, ...)
```

#### 이산 액션 공간
```python
if action_space == "discrete":
    # 액션 토큰화기 사용
    self.action_tokenizer = ActionTokenizer(
        self.tokenizer,
        bins=self.act_head_configs["n_bin"],
        min_action=self.act_head_configs["min_action"],
        max_action=self.act_head_configs["max_action"]
    )
```

#### 학습 아이디어
- **학습 가능한 액션 토큰**: 연속 액션을 위한 특별한 학습 가능한 토큰
- **액션 토큰화**: 이산 액션을 위한 토큰 기반 표현
- **위치 기반 액션 예측**: 액션 토큰의 위치를 통한 액션 예측

### 3. 히스토리 모델링 (History Modeling)

#### 히스토리 타입별 처리
```python
history_type = self.act_head_configs.get("history_type", "post")

if history_type in ["post", "pre"]:
    # 시퀀스 차원 재구성
    vision_x = vision_x.reshape(bs * seq_len, *vision_x.shape[2:]).unsqueeze(1)
    lang_x = lang_x.unsqueeze(1).repeat(1, seq_len, 1).flatten(0, 1)
    
if history_type == "pre":
    # 사전 히스토리 모델링
    multimodal_embeds = rearrange(multimodal_embeds, "(b l) n d -> b (l n) d", l=seq_len)
```

#### 학습 아이디어
- **다단계 관찰 처리**: 과거 관찰을 시퀀스로 처리
- **시간적 인과성**: 시간 순서를 고려한 히스토리 모델링
- **유연한 히스토리 통합**: pre/post/video 등 다양한 히스토리 통합 방식

### 4. 상태 정보 통합 (State Information Integration)

```python
if rel_state is not None and self.use_state:
    # 로봇 상태 인코딩
    state_token = self.encode_state(rel_state)
    state_token = state_token.reshape(bs * seq_len, state_token.shape[-1]).unsqueeze(1)
    
    # 상태 토큰을 멀티모달 시퀀스에 삽입
    multimodal_embeds = self.merge_multi_modal_input(
        multimodal_embeds, state_token, ..., insert_idx=insert_idx
    )
```

#### 학습 아이디어
- **다중 센서 융합**: 비전, 언어, 관절 상태 통합
- **상태 임베딩**: 로봇 상태를 토큰으로 변환
- **위치 기반 상태 삽입**: 상태 정보를 적절한 위치에 삽입

## 학습 파이프라인

### 1. 순전파 과정 (Forward Pass)

#### 연속 액션 학습
```python
def forward_continuous(self, vision_x, lang_x, action_labels, ...):
    # 1. 입력 임베딩
    input_embeds = self.word_embedding(lang_x)
    
    # 2. 멀티모달 융합
    multimodal_embeds = self.merge_multi_modal_input(input_embeds, vision_x, ...)
    
    # 3. 액션 토큰 삽입
    if action_space == "continuous":
        action_tokens = repeat(self.action_token, "d -> b n d", ...)
        multimodal_embeds = self.merge_multi_modal_input(multimodal_embeds, action_tokens, ...)
    
    # 4. VLM 순전파
    output = self.model(inputs_embeds=multimodal_embeds, ...)
    
    # 5. 액션 예측
    action_hs = output.hidden_states[-1][action_token_mask]
    action_logits, action_loss = self._forward_action_head(action_hs, action_labels, ...)
```

#### 이산 액션 학습
```python
def forward_discrete(self, vision_x, lang_x, action_labels, ...):
    # 1. 시퀀스 재구성
    instr_and_action_ids = instr_and_action_ids.flatten(0, 1)
    input_embeds = self.word_embedding(instr_and_action_ids)
    
    # 2. 멀티모달 융합
    multimodal_embeds = self.merge_multi_modal_input(input_embeds, vision_x, ...)
    
    # 3. VLM 순전파
    output = self.model(inputs_embeds=multimodal_embeds, ...)
    
    # 4. 액션 토큰 추출 및 손실 계산
    action_loss = self._forward_action_head(output.logits, action_labels, ...)
```

### 2. 손실 함수 (Loss Functions)

#### 액션 예측 손실
```python
def _forward_action_head(self, action_tokens, action_labels, action_mask):
    # 액션 헤드를 통한 예측
    action = self.act_head(action_tokens, actions=action_labels, action_masks=action_mask)
    
    # 손실 계산
    if action_labels is not None:
        action, action_labels, action_mask = self.act_head.get_labels(action, action_labels, action_mask)
        action_loss = self.act_head.loss(action, action_labels, action_mask)
    
    return action, action_loss
```

#### 멀티태스크 손실
```python
def forward(self, data_source, ...):
    loss = {}
    
    if "action" in data_source:
        # 액션 예측 손실
        action_loss = self.forward_action(...)
        loss = self._update_loss(loss, action_loss)
    
    if "vl_pretrain" in data_source:
        # 비전-언어 사전 훈련 손실
        vl_loss = self.forward_vl_task(...)
        loss = self._update_loss(loss, vl_loss)
    
    return loss
```

### 3. 추론 과정 (Inference)

#### 연속 액션 추론
```python
def inference(self, vision_x, lang_x, ...):
    if action_space == "continuous":
        # 연속 액션 예측
        action_logits = self.forward_continuous(vision_x, lang_x, ..., mode="inference")
        prediction["action"] = action_logits
```

#### 이산 액션 추론
```python
def pred_action_discrete(self, instr_and_action_ids, vision_x, ...):
    # 자동회귀적 액션 생성
    generated_ids = []
    kv_cache = None
    
    for i in range(action_dim * self.fwd_pred_next_n):
        output_hs = self.model(inputs_embeds=multimodal_embeds, past_key_values=kv_cache, use_cache=True)
        kv_cache = output_hs.past_key_values
        cur_id = output_hs.logits[:, -1].argmax(dim=-1)
        generated_ids.append(cur_id)
    
    # 토큰을 액션으로 디코딩
    discretized_actions = self.action_tokenizer.decode_token_ids_to_actions(predicted_action_ids)
    return discretized_actions
```

## 핵심 학습 메커니즘

### 1. 토큰 기반 멀티모달 학습

#### 핵심 아이디어
- **통합 토큰 시퀀스**: 모든 모달리티를 하나의 토큰 시퀀스로 처리
- **위치 기반 삽입**: 모달리티별 토큰을 특정 위치에 삽입
- **어텐션 기반 융합**: 트랜스포머의 셀프 어텐션을 통한 자연스러운 융합

#### 구현 세부사항
```python
# 이미지 토큰 처리
if self.start_image_token_id is not None:
    image_start_embed = self.word_embedding(self.start_image_token_id)
    image_end_embed = self.word_embedding(self.end_image_token_id)
    rgb_feats = torch.cat([image_start_embed, rgb_feats, image_end_embed], dim=2)

# 시퀀스 차원 평탄화
rgb_feats = rearrange(rgb_feats, "b l n d -> b (l n) d")

# 멀티모달 임베딩 결합
multimodal_embeds = torch.cat([input_embeds[:, :insert_idx], rgb_feats, input_embeds[:, insert_idx:]], dim=1)
```

### 2. 액션 토큰 학습

#### 연속 액션 공간
- **학습 가능한 액션 토큰**: `nn.Parameter`로 정의된 특별한 토큰
- **위치 기반 예측**: 액션 토큰의 위치에서 액션 예측
- **마스크 기반 추출**: 액션 토큰만 추출하여 액션 헤드로 전달

#### 이산 액션 공간
- **액션 토큰화**: 연속 액션을 이산 토큰으로 변환
- **자동회귀 생성**: 토큰 기반 자동회귀적 액션 생성
- **디코딩**: 토큰을 다시 연속 액션으로 변환

### 3. 히스토리 모델링

#### 다양한 히스토리 타입
- **post**: 현재 관찰 후 히스토리 처리
- **pre**: 현재 관찰 전 히스토리 처리  
- **video**: 비디오 시퀀스로 히스토리 처리

#### 시간적 모델링
```python
if history_type == "pre":
    # 사전 히스토리 모델링
    multimodal_embeds = rearrange(multimodal_embeds, "(b l) n d -> b (l n) d", l=seq_len)
    output_hs = rearrange(output_hs, "b (l n) d -> (b l) n d", l=seq_len)
```

### 4. 상태 정보 통합

#### 다중 센서 융합
```python
def encode_state(self, state):
    # 관절 상태 임베딩
    arm_state_embeddings = self.embed_arm_state(state[..., :6])
    # 그리퍼 상태 임베딩
    gripper_state_embeddings = self.embed_gripper_state(state[..., [-1]]).long()
    # 상태 임베딩 결합
    state_embeddings = torch.cat((arm_state_embeddings, gripper_state_embeddings), dim=2)
    state_embeddings = self.embed_state(state_embeddings)
    return state_embeddings
```

## 학습 전략

### 1. 멀티태스크 학습

#### 데이터 소스별 학습
```python
def forward(self, data_source, ...):
    if "action" in data_source:
        # 액션 예측 태스크
        action_loss = self.forward_action(...)
    
    if "vl_pretrain" in data_source:
        # 비전-언어 사전 훈련 태스크
        vl_loss = self.forward_vl_task(...)
```

### 2. 점진적 학습

#### 백본 동결/해제
```python
if self.train_setup_configs["freeze_backbone"]:
    model.requires_grad_(False)
else:
    if self.train_setup_configs.get("train_decoder_layers", -1) == -1:
        model.requires_grad_(True)
    else:
        # 특정 레이어만 훈련
        for layer in self.text_tower.layers[-self.train_setup_configs["train_decoder_layers"]:]:
            layer.requires_grad_(True)
```

### 3. 효율적 훈련

#### LoRA 적응
```python
if self.train_setup_configs["lora_enable"]:
    lora_config = LoraConfig(
        r=self.train_setup_configs["lora_r"],
        lora_alpha=self.train_setup_configs["lora_alpha"],
        target_modules=find_all_linear_names(model),
        lora_dropout=self.train_setup_configs["lora_dropout"],
        bias=self.train_setup_configs["lora_bias"],
        task_type="CAUSAL_LM"
    )
    self.model = get_peft_model(model, lora_config)
```

## 결론

RoboVLMs의 핵심 학습 아이디어는 다음과 같습니다:

### 주요 학습 메커니즘
1. **토큰 기반 멀티모달 융합**: 모든 모달리티를 통합된 토큰 시퀀스로 처리
2. **학습 가능한 액션 토큰**: 연속/이산 액션 공간을 위한 특별한 토큰 학습
3. **유연한 히스토리 모델링**: 다양한 히스토리 통합 방식 지원
4. **다중 센서 융합**: 비전, 언어, 관절 상태의 통합 학습

### 학습 파이프라인 특징
1. **멀티태스크 학습**: 액션 예측과 비전-언어 사전 훈련 동시 학습
2. **점진적 학습**: 백본 동결/해제를 통한 효율적 학습
3. **효율적 적응**: LoRA를 통한 파라미터 효율적 미세 조정
4. **유연한 추론**: 연속/이산 액션 공간 모두 지원

이러한 설계를 통해 RoboVLMs는 VLM의 강력한 멀티모달 표현 능력을 로봇 조작에 효과적으로 활용할 수 있습니다.
