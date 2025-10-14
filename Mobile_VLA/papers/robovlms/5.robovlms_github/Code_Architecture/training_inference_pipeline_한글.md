# RoboVLMs 훈련/추론 파이프라인 분석 (한글)

## 훈련 파이프라인 개요

RoboVLMs의 `BaseRoboVLM` 코드를 통해 훈련과 추론의 전체 파이프라인을 분석합니다. 이 파이프라인은 VLM을 VLA로 변환하는 핵심 과정을 보여줍니다.

## 1. 훈련 파이프라인 (Training Pipeline)

### 1.1 연속 액션 훈련 (Continuous Action Training)

#### 순전파 과정
```python
def forward_continuous(self, vision_x, lang_x, action_labels, ...):
    # 1. 입력 처리
    bs, seq_len = vision_x.shape[:2]
    action_space = self.act_head_configs.get("action_space", "continuous")
    
    # 2. 히스토리 타입별 처리
    history_type = self.act_head_configs.get("history_type", "post")
    if history_type in ["post", "pre"]:
        vision_x = vision_x.reshape(bs * seq_len, *vision_x.shape[2:]).unsqueeze(1)
        lang_x = lang_x.unsqueeze(1).repeat(1, seq_len, 1).flatten(0, 1)
    
    # 3. 언어 임베딩
    input_embeds = self.word_embedding(lang_x)
    
    # 4. 멀티모달 융합
    multimodal_embeds = self.merge_multi_modal_input(
        input_embeds, vision_x, labels=None, attention_mask=attention_mask, insert_idx=bos_offset
    )
    
    # 5. 상태 정보 통합 (선택적)
    if rel_state is not None and self.use_state:
        state_token = self.encode_state(rel_state)
        multimodal_embeds = self.merge_multi_modal_input(
            multimodal_embeds, state_token, ..., insert_idx=insert_idx
        )
    
    # 6. 액션 토큰 삽입
    if action_space == "continuous":
        action_tokens = repeat(self.action_token, "d -> b n d", 
                              b=multimodal_embeds.shape[0], n=self.latent_num)
        multimodal_embeds = self.merge_multi_modal_input(
            multimodal_embeds, action_tokens, ..., insert_idx=insert_idx
        )
    
    # 7. VLM 순전파
    output = self.model(
        input_ids=None,
        attention_mask=multimodal_attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=multimodal_embeds,
        use_cache=use_cache,
        output_hidden_states=True
    )
    
    # 8. 액션 예측
    output_hs = output.hidden_states[-1].clone()
    action_hs = output_hs[action_token_mask].reshape(bs, seq_len, self.latent_num, -1)
    action_logits, action_loss = self._forward_action_head(action_hs, action_labels, action_mask)
    
    return action_logits, action_loss
```

#### 손실 계산
```python
def _forward_action_head(self, action_tokens, action_labels, action_mask):
    # 액션 헤드를 통한 예측
    action = self.act_head(action_tokens, actions=action_labels, action_masks=action_mask)
    
    # 손실 계산
    if action_labels is not None:
        action, action_labels, action_mask = self.act_head.get_labels(
            action, action_labels, action_mask, tok_seq=action_tokens
        )
        action_loss = self.act_head.loss(action, action_labels, action_mask)
    
    return action, action_loss
```

### 1.2 이산 액션 훈련 (Discrete Action Training)

#### 순전파 과정
```python
def forward_discrete(self, vision_x, lang_x, action_labels, ...):
    # 1. 시퀀스 재구성
    bs, window_size = vision_x.shape[:2]
    instr_and_action_ids = instr_and_action_ids.flatten(0, 1)
    instr_and_action_labels = instr_and_action_labels.flatten(0, 1)
    instr_and_action_mask = instr_and_action_mask.flatten(0, 1)
    
    # 2. 언어 임베딩
    input_embeds = self.word_embedding(instr_and_action_ids)
    vision_x = vision_x.flatten(0, 1)
    
    # 3. 멀티모달 융합
    multimodal_embeds = self.merge_multi_modal_input(
        input_embeds, vision_x, instr_and_action_labels, instr_and_action_mask
    )
    
    # 4. 시퀀스 차원 복원
    multimodal_embeds = rearrange(
        multimodal_embeds, "(bs ws) seq_len ... -> bs (ws seq_len) ...", bs=bs, ws=window_size
    )
    
    # 5. VLM 순전파
    output = self.model(
        input_ids=None,
        attention_mask=multimodal_attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=multimodal_embeds,
        use_cache=use_cache
    )
    
    # 6. 액션 예측
    output_hs = output.logits
    _, action_loss = self._forward_action_head(output_hs, instr_and_action_labels, multimodal_attention_mask)
    
    return action_loss
```

### 1.3 멀티태스크 훈련 (Multi-task Training)

#### 통합 훈련 과정
```python
def forward(self, data_source, ...):
    loss = {}
    
    # 액션 예측 태스크
    if "action" in data_source:
        action_loss = self.forward_action(
            vision_x=vision_x,
            lang_x=lang_x,
            attention_mask=attention_mask,
            action_labels=action_labels,
            action_mask=action_mask,
            ...
        )
        loss = self._update_loss(loss, action_loss)
    
    # 비전-언어 사전 훈련 태스크
    if "vl_pretrain" in data_source:
        vl_loss = self.forward_vl_task(
            input_ids=instr_and_action_ids,
            labels=instr_and_action_labels,
            attention_mask=instr_and_action_mask,
            images=vision_x
        )
        loss = self._update_loss(loss, vl_loss)
    
    return loss
```

## 2. 추론 파이프라인 (Inference Pipeline)

### 2.1 연속 액션 추론 (Continuous Action Inference)

#### 추론 과정
```python
def inference(self, vision_x, lang_x, ...):
    prediction = {}
    
    if self.train_setup_configs["predict_action"]:
        if action_space == "continuous":
            # 연속 액션 예측
            action_logits = self.forward_continuous(
                vision_x, lang_x, attention_mask, vision_gripper=vision_gripper, mode="inference"
            )
            prediction["action"] = action_logits
        elif action_space == "discrete":
            # 이산 액션 예측
            action = self.pred_action_discrete(lang_x, vision_x, vision_gripper, attention_mask)
            prediction["action"] = action
    
    return prediction
```

### 2.2 이산 액션 추론 (Discrete Action Inference)

#### 자동회귀적 액션 생성
```python
def pred_action_discrete(self, instr_and_action_ids, vision_x, vision_gripper=None, attention_mask=None):
    # 1. 멀티모달 입력 준비
    input_embeds = self.word_embedding(instr_and_action_ids)
    multimodal_embeds = self.merge_multi_modal_input(input_embeds, vision_x, attention_mask=attention_mask)
    
    if vision_gripper is not None:
        multimodal_embeds = self.merge_multi_modal_input(
            multimodal_embeds, vision_gripper, attention_mask=multimodal_attention_mask
        )
    
    # 2. 자동회귀적 액션 생성
    action_dim = self.act_head_configs["action_dim"]
    generated_ids = []
    kv_cache = None
    
    for i in range(action_dim * self.fwd_pred_next_n):
        if kv_cache is None:
            output_hs = self.model(
                inputs_embeds=multimodal_embeds,
                past_key_values=kv_cache,
                use_cache=True
            )
        else:
            output_hs = self.model(
                inputs_embeds=multimodal_embeds[:, -1:],
                past_key_values=kv_cache,
                use_cache=True
            )
        
        kv_cache = output_hs.past_key_values
        cur_id = output_hs.logits[:, -1].argmax(dim=-1)
        generated_ids.append(cur_id)
        
        # 다음 토큰을 위한 임베딩 추가
        cur_embed = self.word_embedding(cur_id)
        multimodal_embeds = torch.cat([multimodal_embeds, cur_embed.unsqueeze(1)], dim=1)
    
    # 3. 토큰을 액션으로 디코딩
    generated_ids = torch.cat(generated_ids, dim=0).reshape(self.fwd_pred_next_n, action_dim)
    predicted_action_ids = generated_ids[:, -action_dim:].cpu().numpy()
    discretized_actions = self.action_tokenizer.decode_token_ids_to_actions(predicted_action_ids)
    
    # 4. 액션 후처리
    if isinstance(discretized_actions, list):
        discretized_actions = np.array(discretized_actions)
    discretized_actions[:, -1] = np.where(discretized_actions[:, -1] > 0, 1, -1)
    
    return discretized_actions
```

## 3. 핵심 파이프라인 구성요소

### 3.1 멀티모달 입력 융합

#### 이미지 인코딩
```python
def encode_images(self, images, image_sizes=None):
    # 입력 이미지 처리
    if images.ndim == 4:
        images = images.unsqueeze(1)
    
    bs, seq_len = images.shape[:2]
    
    # 이미지 특징 추출
    if type(images) is list or images.ndim == 5:
        concat_images = torch.cat([image for image in images], dim=0)
        image_features = self.model_encode_images(concat_images)
        # 분할된 특징 재구성
        split_sizes = [image.shape[0] for image in images]
        image_features = torch.split(image_features, split_sizes, dim=0)
        image_features = [x.flatten(0, 1) for x in image_features]
    else:
        image_features = self.model_encode_images(images)
    
    # 특징 스택 및 리샘플링
    image_features = torch.stack(image_features, dim=0).view(bs, seq_len, -1, image_features[0].shape[-1])
    
    if self.use_vision_resampler:
        image_features = self.vision_resampler(image_features.unsqueeze(2))
    
    return image_features
```

#### 멀티모달 융합
```python
def merge_multi_modal_input(self, input_embeds, multimodal_feats, ...):
    # 이미지 특징 처리
    if is_image:
        rgb_feats = self.encode_images(multimodal_feats)
        
        # 이미지 토큰에 시작/끝 토큰 추가
        if self.start_image_token_id is not None:
            image_start_embed = self.word_embedding(self.start_image_token_id)
            image_end_embed = self.word_embedding(self.end_image_token_id)
            rgb_feats = torch.cat([image_start_embed, rgb_feats, image_end_embed], dim=2)
        
        # 시퀀스 차원 평탄화
        rgb_feats = rearrange(rgb_feats, "b l n d -> b (l n) d")
    else:
        rgb_feats = multimodal_feats
    
    # 멀티모달 임베딩 결합
    multimodal_embeds = torch.cat([
        input_embeds[:, :insert_idx], 
        rgb_feats, 
        input_embeds[:, insert_idx:]
    ], dim=1)
    
    # 어텐션 마스크 및 라벨 처리
    insert_mask = torch.cat([
        torch.zeros(input_embeds[:, :insert_idx].shape[:2]),
        torch.ones(rgb_feats.shape[:2]),
        torch.zeros(input_embeds[:, insert_idx:].shape[:2])
    ], dim=1).bool().to(multimodal_embeds.device)
    
    return multimodal_embeds, mutlimodal_labels, multimodal_attention_mask, insert_mask
```

### 3.2 액션 예측 헤드

#### 액션 헤드 순전파
```python
def _forward_action_head(self, action_tokens, action_labels, action_mask):
    # 액션 헤드를 통한 예측
    action = self.act_head(
        action_tokens, 
        actions=action_labels, 
        action_masks=action_mask
    )
    
    # 손실 계산
    action_loss = None
    if action_labels is not None:
        action, action_labels, action_mask = self.act_head.get_labels(
            action, action_labels, action_mask, tok_seq=action_tokens
        )
        action_loss = self.act_head.loss(action, action_labels, action_mask)
    
    return action, action_loss
```

### 3.3 상태 정보 통합

#### 상태 인코딩
```python
def encode_state(self, state):
    # 관절 상태 임베딩 (첫 6개 차원)
    arm_state_embeddings = self.embed_arm_state(state[..., :6])
    # 그리퍼 상태 임베딩 (마지막 차원)
    gripper_state_embeddings = self.embed_gripper_state(state[..., [-1]]).long()
    # 상태 임베딩 결합
    state_embeddings = torch.cat((arm_state_embeddings, gripper_state_embeddings), dim=2)
    state_embeddings = self.embed_state(state_embeddings)
    return state_embeddings
```

## 4. 훈련 전략

### 4.1 점진적 학습

#### 백본 동결/해제
```python
def _trainable_params_setup(self):
    model = self.model
    
    # 백본 동결/해제
    if self.train_setup_configs["freeze_backbone"]:
        model.requires_grad_(False)
    else:
        if self.train_setup_configs.get("train_decoder_layers", -1) == -1:
            model.requires_grad_(True)
        else:
            # 특정 레이어만 훈련
            for layer in self.text_tower.layers[-self.train_setup_configs["train_decoder_layers"]:]:
                layer.requires_grad_(True)
    
    # 비전 타워 훈련 제어
    if self.train_setup_configs.get("train_vision", False):
        self.vision_tower.requires_grad_(True)
    else:
        self.vision_tower.requires_grad_(False)
```

### 4.2 LoRA 적응

#### LoRA 설정
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

### 4.3 그래디언트 체크포인팅

#### 메모리 효율적 훈련
```python
if hasattr(model, "enable_input_require_grads"):
    model.enable_input_require_grads()
    model.gradient_checkpointing = True
    model.training = True
else:
    def make_inputs_require_grad(module, input, output):
        output.requires_grad_(True)
    self.word_embedding.register_forward_hook(make_inputs_require_grad)
```

## 5. 추론 최적화

### 5.1 캐시 기반 추론

#### KV 캐시 활용
```python
def pred_action_discrete(self, ...):
    generated_ids = []
    kv_cache = None
    
    for i in range(action_dim * self.fwd_pred_next_n):
        if kv_cache is None:
            # 첫 번째 토큰 생성
            output_hs = self.model(
                inputs_embeds=multimodal_embeds,
                past_key_values=kv_cache,
                use_cache=True
            )
        else:
            # 이전 캐시 활용
            output_hs = self.model(
                inputs_embeds=multimodal_embeds[:, -1:],
                past_key_values=kv_cache,
                use_cache=True
            )
        
        kv_cache = output_hs.past_key_values
        cur_id = output_hs.logits[:, -1].argmax(dim=-1)
        generated_ids.append(cur_id)
```

### 5.2 배치 처리

#### 효율적 배치 추론
```python
def inference(self, vision_x, lang_x, ...):
    # 배치 차원 처리
    bs, seq_len = vision_x.shape[:2]
    
    # 배치별 추론
    predictions = []
    for i in range(bs):
        pred = self._single_inference(vision_x[i], lang_x[i], ...)
        predictions.append(pred)
    
    return predictions
```

## 결론

RoboVLMs의 훈련/추론 파이프라인은 다음과 같은 특징을 가집니다:

### 훈련 파이프라인 특징
1. **멀티모달 융합**: 비전, 언어, 액션을 통합된 토큰 시퀀스로 처리
2. **유연한 액션 공간**: 연속/이산 액션 공간 모두 지원
3. **멀티태스크 학습**: 액션 예측과 비전-언어 사전 훈련 동시 수행
4. **점진적 학습**: 백본 동결/해제를 통한 효율적 학습
5. **효율적 적응**: LoRA를 통한 파라미터 효율적 미세 조정

### 추론 파이프라인 특징
1. **자동회귀적 생성**: 이산 액션 공간에서 토큰 기반 생성
2. **캐시 기반 추론**: KV 캐시를 활용한 효율적 추론
3. **배치 처리**: 효율적인 배치 추론 지원
4. **유연한 모드**: 훈련/추론 모드 자동 전환

이러한 파이프라인을 통해 RoboVLMs는 VLM의 강력한 멀티모달 표현 능력을 로봇 조작에 효과적으로 활용할 수 있습니다.
