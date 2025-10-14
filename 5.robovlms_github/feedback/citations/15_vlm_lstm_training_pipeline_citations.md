# 15. VLM Fine-tuning과 LSTM Layer 학습 파이프라인 상세 분석

## 📋 개요

이 문서는 RoboVLMs에서 VLM Fine-tuning과 LSTM Layer 학습 과정을 일반적인 AI 학습 파이프라인 방식으로 자세히 설명합니다.

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
    "gradient_checkpointing": False # 그래디언트 체크포인팅 비활성화
}
```

**출처**: `RoboVLMs/README.md:228-250`

**LoRA 설정 (선택적)**
```python
# LoRA 설정 (일부 모델에서 사용)
lora_config = {
    "lora_enable": True,           # LoRA 활성화
    "lora_r": 64,                  # LoRA rank
    "lora_alpha": 16,              # LoRA alpha
    "lora_dropout": 0.05,         # LoRA 드롭아웃
    "lora_bias": "none"            # LoRA bias 설정
}
```

### 2.3 VLM 학습 과정

**멀티모달 입력 처리**
```python
# BaseRoboVLM.forward()에서 멀티모달 처리
def forward(self, vision_x, lang_x, attention_mask=None, **kwargs):
    # 1단계: 비전 인코딩
    vision_features = self.encode_images(vision_x)
    
    # 2단계: 텍스트 인코딩
    text_features = self.encode_text(lang_x)
    
    # 3단계: 멀티모달 융합
    multimodal_features = self.merge_multi_modal_input(
        vision_features, text_features
    )
    
    # 4단계: VLM Forward Pass
    output = self.model(
        input_ids=multimodal_features,
        attention_mask=attention_mask,
        output_hidden_states=True
    )
```

**출처**: `RoboVLMs/robovlms/model/backbone/base_backbone.py:1261-1284`

## 🧠 3. LSTM Layer 학습 과정

### 3.1 LSTM Decoder 아키텍처

**LSTM Decoder 구조**
```python
class LSTMDecoder(BasePolicyHead):
    def __init__(self, in_features, action_dim, down_sample, latent, 
                 fwd_pred_next_n, window_size, hidden_size=1024, num_layers=4):
        
        # LSTM 레이어 초기화
        self.rnn = lstm_decoder(
            in_features * latent,      # 입력 차원
            hidden_size * latent,      # 히든 차원
            num_layers,                # LSTM 레이어 수
            policy_rnn_dropout_p=0.0   # 드롭아웃 비율
        )
        
        # 액션 헤드 (팔 액션용)
        self.actions = MLPTanhHead(
            self.hidden_size * latent, 
            fwd_pred_next_n * (action_dim - 1)  # 6-DOF 팔 액션
        )
        
        # 그리퍼 헤드 (그리퍼 액션용)
        self.gripper = MLPSigmoidHead(
            self.hidden_size * latent, 
            fwd_pred_next_n  # 1-DOF 그리퍼 액션
        )
```

**출처**: `RoboVLMs/robovlms/model/policy_head/base_policy.py:173-204`

### 3.2 LSTM 학습 과정

**1단계: VLM 특징 추출**
```python
# VLM에서 추출된 특징을 LSTM 입력으로 사용
def forward(self, tok_seq, h_0=None, **kwargs):
    # VLM 출력 특징: [batch_size, window_size, latent, feature_dim]
    # LSTM 입력으로 변환: [batch_size, window_size, latent * feature_dim]
    
    if self.down_sample == "none":
        tok_seq = rearrange(tok_seq, "b l n d-> b l (n d)")
    
    # LSTM Forward Pass
    x, h_n = self.rnn(tok_seq, self.hidden_state)
    self.hidden_state = h_n
```

**출처**: `RoboVLMs/robovlms/model/policy_head/base_policy.py:223-224`

**2단계: 액션 예측**
```python
# LSTM 출력을 액션으로 변환
def forward(self, tok_seq, **kwargs):
    # LSTM 처리
    x, h_n = self.rnn(tok_seq, self.hidden_state)
    
    # 액션 예측
    actions = self.actions(x)      # 팔 액션 (6-DOF)
    gripper = self.gripper(x)      # 그리퍼 액션 (1-DOF)
    
    # 출력 형태 조정
    actions = rearrange(actions, "b l (n d) -> b l n d", n=self.fwd_pred_next_n)
    gripper = rearrange(gripper, "b l (n d) -> b l n d", n=self.fwd_pred_next_n)
    
    return actions, gripper
```

**출처**: `RoboVLMs/robovlms/model/policy_head/base_policy.py:223-224`

### 3.3 Loss 계산 및 학습

**Loss 계산 과정**
```python
def loss(self, pred_action_logits, labels, attention_mask=None):
    # 1단계: 시퀀스 시프트 (autoregressive 학습)
    shift_logits = pred_action_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # 2단계: CrossEntropyLoss 계산
    loss_fct = nn.CrossEntropyLoss()
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)), 
        shift_labels.view(-1)
    )
    
    # 3단계: 액션 마스킹
    mask = torch.logical_and(
        labels > self.action_tokenizer.action_token_begin_idx,
        labels < self.action_tokenizer.action_token_end_idx
    )
    
    # 4단계: 정확도 계산
    pred_action = pred_action_logits.argmax(dim=-1)
    correct_preds = torch.logical_and((pred_action == labels), mask)
    
    return {
        "loss_arm": loss,
        "acc_arm": arm_acc,
        "acc_gripper": gripper_acc
    }
```

**출처**: `RoboVLMs/robovlms/model/policy_head/base_policy.py:226-281`

## 🔄 4. 전체 학습 파이프라인

### 4.1 학습 단계별 과정

**1단계: 데이터 로딩**
```python
# BaseTrainer.training_step()에서 배치 처리
def training_step(self, batch, batch_idx):
    # 배치 데이터 추출
    (rgb, hand_rgb, attention_mask, language, text_mask,
     arm_action, gripper_action, instr_and_action_ids,
     instr_and_action_labels, instr_and_action_mask) = self._process_batch(batch)
```

**출처**: `RoboVLMs/robovlms/train/base_trainer.py:565-591`

**2단계: 모델 Forward Pass**
```python
# 모델 Forward Pass
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
```

**출처**: `RoboVLMs/robovlms/train/base_trainer.py:593-619`

**3단계: Loss 계산**
```python
# Loss 계산 및 최적화
def _get_loss(self, prediction):
    loss_arm_act = prediction.get("loss_arm_act", None)
    loss_gripper_act = prediction.get("loss_gripper_act", None)
    
    # 액션 Loss 계산
    loss_act = (loss_arm_act if loss_arm_act is not None else 0) + (
        loss_gripper_act * self.arm_gripper_loss_ratio
        if loss_gripper_act is not None else 0
    )
    
    return {"loss": loss_act, "loss_act": loss_act}
```

**출처**: `RoboVLMs/robovlms/train/base_trainer.py:269-315`

### 4.2 학습 설정

**하이퍼파라미터**
```python
# 학습 설정
training_config = {
    "learning_rate": 1e-4,           # 학습률
    "weight_decay": 0.0,             # 가중치 감쇠
    "batch_size": 4,                 # 배치 크기
    "max_epochs": 5,                 # 최대 에포크
    "gradient_clip_val": 1.0,        # 그래디언트 클리핑
    "precision": "bf16"               # 혼합 정밀도
}
```

**출처**: `RoboVLMs/README.md:221-223`

**Loss 가중치**
```python
# Loss 가중치 설정
loss_weights = {
    "arm_gripper_loss_ratio": 0.01,   # 팔-그리퍼 Loss 비율
    "cap_loss_ratio": 0.05,           # 캡션 Loss 비율
    "fwd_loss_ratio": 0               # 미래 예측 Loss 비율
}
```

**출처**: `RoboVLMs/robovlms/train/base_trainer.py:288-292`

## 🔧 5. RoboVLMs만의 고유한 디테일

### 5.1 멀티모달 융합 메커니즘

**Vision + Language + Action 통합 처리**
```python
# BaseRoboVLM.merge_multi_modal_input()에서 멀티모달 융합
def merge_multi_modal_input(self, input_embeds, vision_x, attention_mask=None):
    # 1단계: 비전 특징 인코딩
    vision_features = self.encode_images(vision_x)
    
    # 2단계: 시작/끝 이미지 토큰 삽입
    start_img_token = self.start_img_token.unsqueeze(0).unsqueeze(0)
    end_img_token = self.end_img_token.unsqueeze(0).unsqueeze(0)
    
    # 3단계: 멀티모달 임베딩 결합
    multimodal_embeds = torch.cat([
        input_embeds[:, :start_idx], 
        start_img_token, 
        vision_features, 
        end_img_token, 
        input_embeds[:, start_idx:]
    ], dim=1)
    
    return multimodal_embeds, attention_mask
```

**출처**: `RoboVLMs/robovlms/model/backbone/base_backbone.py:1390-1410`

### 5.2 이산 액션 생성 과정

**Autoregressive 액션 토큰 생성**
```python
# BaseRoboVLM.pred_action_discrete()에서 이산 액션 생성
def pred_action_discrete(self, instr_and_action_ids, vision_x, vision_gripper=None):
    # 1단계: 멀티모달 융합
    multimodal_embeds = self.merge_multi_modal_input(
        input_embeds, vision_x, attention_mask=attention_mask
    )
    
    # 2단계: 그리퍼 비전 추가 (선택적)
    if vision_gripper is not None:
        multimodal_embeds = self.merge_multi_modal_input(
            multimodal_embeds, vision_gripper, attention_mask=multimodal_attention_mask
        )
    
    # 3단계: Autoregressive 액션 생성
    generated_ids = []
    kv_cache = None
    for i in range(action_dim * self.fwd_pred_next_n):
        output_hs = self.model(
            inputs_embeds=multimodal_embeds,
            past_key_values=kv_cache,
            use_cache=True
        )
        kv_cache = output_hs.past_key_values
        cur_id = output_hs.logits[:, -1].argmax(dim=-1)
        generated_ids.append(cur_id)
        
        # 다음 토큰을 위한 임베딩 추가
        cur_embed = self.word_embedding(cur_id)
        multimodal_embeds = torch.cat([multimodal_embeds, cur_embed.unsqueeze(1)], dim=1)
```

**출처**: `RoboVLMs/robovlms/model/backbone/base_backbone.py:1384-1436`

### 5.3 Vision Resampler (선택적)

**PerceiverResampler를 통한 비전 토큰 압축**
```python
# Vision Resampler 설정
vision_resampler_configs = {
    "use_vision_resampler": True,    # Vision Resampler 사용 여부
    "vision_resampler_configs": {
        "depth": 8,                  # PerceiverResampler 깊이
        "heads": 8,                  # 어텐션 헤드 수
        "dim_head": 64,              # 헤드 차원
        "num_latents": 64            # 압축된 토큰 수 (196 → 64)
    }
}
```

**출처**: `RoboVLMs/robovlms/model/README.md:51-52`

### 5.4 액션 히스토리 관리

**Window Size 기반 히스토리 관리**
```python
# LSTMDecoder에서 히스토리 관리
def forward(self, tok_seq, h_0=None, **kwargs):
    if tok_seq.shape[1] == 1:
        self.history_memory.append(tok_seq)
        if len(self.history_memory) <= self.history_len:
            # 히스토리 길이 내에서 LSTM 처리
            x, h_n = self.rnn(tok_seq, self.hidden_state)
            self.hidden_state = h_n
        else:
            # 히스토리 길이 초과 시 윈도우 슬라이딩
            cur_len = len(self.history_memory)
            for _ in range(cur_len - self.history_len):
                self.history_memory.pop(0)
            
            # 윈도우 크기만큼 히스토리 재구성
            hist_feature = torch.cat(self.history_memory, dim=1)
            self.hidden_state = None
            x, h_n = self.rnn(hist_feature, self.hidden_state)
```

**출처**: `RoboVLMs/robovlms/model/policy_head/base_policy.py:223-224`

### 5.5 액션 임베딩 시스템

**LinearActionEncoder를 통한 액션 인코딩**
```python
# LinearActionEncoder에서 액션 임베딩
class LinearActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size):
        self.arm_mlp = nn.Sequential(
            nn.Linear(action_dim - 1, hidden_size),  # 팔 액션 (6-DOF)
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.gripper_mlp = nn.Sequential(
            nn.Linear(1, hidden_size),               # 그리퍼 액션 (1-DOF)
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
    
    def forward(self, action):
        arm_action = action[:, :, :6]      # 팔 액션
        gripper_action = action[:, :, 6:7] # 그리퍼 액션
        
        arm_embed = self.arm_mlp(arm_action)
        gripper_embed = self.gripper_mlp(gripper_action)
        
        action_embed = torch.cat([arm_embed, gripper_embed], dim=-1)
        return action_embed
```

**출처**: `RoboVLMs/robovlms/model/action_encoder/linear_encoder.py:4-41`

### 5.6 다중 카메라 시스템

**정적 카메라 + 그리퍼 카메라 처리**
```python
# BaseRoboVLM에서 다중 카메라 처리
def forward(self, vision_x, lang_x, vision_gripper=None, **kwargs):
    # 1단계: 정적 카메라 처리
    vision_features = self.encode_images(vision_x)
    
    # 2단계: 그리퍼 카메라 처리 (선택적)
    if vision_gripper is not None:
        gripper_features = self.encode_images(vision_gripper)
        # 그리퍼 특징을 정적 카메라 특징과 결합
        vision_features = torch.cat([vision_features, gripper_features], dim=1)
    
    # 3단계: 멀티모달 융합
    multimodal_features = self.merge_multi_modal_input(
        vision_features, lang_x
    )
```

**출처**: `RoboVLMs/robovlms/model/backbone/base_backbone.py:1399-1409`

### 5.7 액션 토큰 삽입 메커니즘

**⚠️ 중요: 학습 시에만 사용되는 메커니즘**

**학습 시: 액션 토큰을 텍스트 시퀀스에 삽입**
```python
# 학습 시 액션 토큰 삽입 로직 (ActionPredictionBatchTransform)
def cat_input_ids_and_action_ids(self, input_ids, action_ids, eos_token_id, right_pad_len):
    # 1단계: 액션 토큰을 텍스트 시퀀스에 삽입
    input_ids = input_ids + action_ids
    
    # 2단계: 라벨 생성 (액션 토큰 부분만 학습 대상)
    labels = [-100] * len(input_ids[:-len(action_ids)]) + action_ids
    
    # 3단계: 어텐션 마스크 생성
    attention_masks = [1] * len(input_ids)
    
    return input_ids, labels, attention_masks
```

**출처**: `RoboVLMs/robovlms/data/base_action_prediction_dataset.py:141-172`

**추론 시: 액션 토큰 삽입 없이 직접 생성**
```python
# 추론 시 액션 생성 (BaseRoboVLM.pred_action_discrete)
def pred_action_discrete(self, instr_and_action_ids, vision_x, vision_gripper=None):
    # 1단계: 멀티모달 융합 (액션 토큰 삽입 없음)
    multimodal_embeds = self.merge_multi_modal_input(
        input_embeds, vision_x, attention_mask=attention_mask
    )
    
    # 2단계: Autoregressive 액션 생성
    generated_ids = []
    for i in range(action_dim * self.fwd_pred_next_n):
        output_hs = self.model(
            inputs_embeds=multimodal_embeds,
            past_key_values=kv_cache,
            use_cache=True
        )
        cur_id = output_hs.logits[:, -1].argmax(dim=-1)
        generated_ids.append(cur_id)
        
        # 생성된 토큰을 다음 입력으로 사용
        cur_embed = self.word_embedding(cur_id)
        multimodal_embeds = torch.cat([multimodal_embeds, cur_embed.unsqueeze(1)], dim=1)
    
    # 3단계: 액션 토큰을 연속 액션으로 디코딩
    discretized_actions = self.action_tokenizer.decode_token_ids_to_actions(
        predicted_action_ids
    )
    
    return discretized_actions
```

**출처**: `RoboVLMs/robovlms/model/backbone/base_backbone.py:1384-1452`

**학습 vs 추론 차이점**

| 구분 | 학습 시 | 추론 시 |
|------|---------|---------|
| **액션 토큰 삽입** | ✅ 텍스트에 액션 토큰 삽입 | ❌ 액션 토큰 삽입 없음 |
| **목적** | 액션 토큰 예측 학습 | 액션 토큰 자동 생성 |
| **입력** | 텍스트 + 액션 토큰 | 텍스트만 |
| **출력** | 액션 토큰 예측 | 액션 토큰 생성 |
| **방식** | Teacher Forcing | Autoregressive Generation |

## 🔧 6. 실제 FT 코드와 LSTM Layer 학습 코드

### 6.1 VLM Fine-tuning 코드

**BaseRoboVLM._trainable_params_setup() - 파라미터 동결 설정**
```python
def _trainable_params_setup(self):
    model = self.model
    
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
    
    # 2단계: 비전 타워 동결 설정
    if self.train_setup_configs.get("train_vision", False):
        self.vision_tower.requires_grad_(True)
    else:
        self.vision_tower.requires_grad_(False)
    
    # 3단계: LoRA 설정
    if self.train_setup_configs["lora_enable"]:
        # LoRA 파라미터만 학습 가능하도록 설정
        pass
```

**출처**: `RoboVLMs/robovlms/model/backbone/base_backbone.py:470-512`

**BaseTrainer.training_step() - 학습 스텝**
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

### 6.2 LSTM Layer 학습 코드

**LSTMDecoder.forward() - LSTM Forward Pass**
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

**LSTMDecoder.loss() - Loss 계산**
```python
def loss(self, pred_action_logits, labels, attention_mask=None):
    # 1단계: 시퀀스 시프트 (autoregressive 학습)
    shift_logits = pred_action_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # 2단계: CrossEntropyLoss 계산
    loss_fct = nn.CrossEntropyLoss()
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)), 
        shift_labels.view(-1)
    )
    
    # 3단계: 액션 마스킹
    mask = torch.logical_and(
        labels > self.action_tokenizer.action_token_begin_idx,
        labels < self.action_tokenizer.action_token_end_idx
    )
    
    # 4단계: 정확도 계산
    pred_action = pred_action_logits.argmax(dim=-1)
    correct_preds = torch.logical_and((pred_action == labels), mask)
    
    # 5단계: 팔/그리퍼 정확도 분리 계산
    arm_acc = correct_preds_cut[:, :6].sum().float() / correct_preds_cut[:, :6].numel()
    gripper_acc = correct_preds_cut[:, -1].sum().float() / correct_preds_cut[:, -1].numel()
    
    return {
        "loss_arm": loss,
        "acc_arm": arm_acc,
        "acc_gripper": gripper_acc
    }
```

**출처**: `RoboVLMs/robovlms/model/policy_head/base_policy.py:226-281`

### 6.3 실제 학습 루프 코드

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

### 6.4 Loss 계산 함수

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

### 6.5 설정 파일 예시

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

## 🔧 7. 학습 변수와 추론 변수 상세 분석

### 7.1 학습 변수 (Training Variables)

**BaseRoboVLM._trainable_params_setup() - 학습 가능한 파라미터 설정**
```python
def _trainable_params_setup(self):
    model = self.model
    
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
    
    # 2단계: 비전 타워 동결 설정
    if self.train_setup_configs.get("train_vision", False):
        self.vision_tower.requires_grad_(True)
    else:
        self.vision_tower.requires_grad_(False)
    
    # 3단계: LoRA 설정
    if self.train_setup_configs["lora_enable"]:
        # LoRA 파라미터만 학습 가능하도록 설정
        pass
```

**출처**: `RoboVLMs/robovlms/model/backbone/base_backbone.py:470-512`

**BaseTrainer.get_grouped_params() - 학습 파라미터 그룹화**
```python
def get_grouped_params(self, model):
    return [
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad],
            "weight_decay": self.configs["weight_decay"],
        }
    ]
```

**출처**: `RoboVLMs/robovlms/train/base_trainer.py:716-722`

**RoboFlamingo._trainable_params_setup() - Flamingo 모델 학습 설정**
```python
def _trainable_params_setup(self):
    self.requires_grad_(False)
    
    # 1단계: 비전 인코더 학습 설정
    if self.train_setup_configs["train_vision"]:
        self.vision_encoder.requires_grad_(True)
    
    # 2단계: 디코더 레이어 학습 설정
    if self.train_setup_configs["train_decoder_layers"] == -1:
        self.model.gated_cross_attn_layers.requires_grad_(True)
    else:
        # 마지막 N개 레이어만 학습
        ix = self.train_setup_configs["train_decoder_layers"]
        for layer in self.model.gated_cross_attn_layers[-ix:]:
            layer.requires_grad_(True)
    
    # 3단계: 전체 디코더 학습 설정
    if self.train_setup_configs["train_full_decoder"]:
        self.model.requires_grad_(True)
    
    # 4단계: 리샘플러 학습 설정
    if self.train_setup_configs["train_resampler"]:
        self.perceiver.requires_grad_(True)
    else:
        self.perceiver.requires_grad_(False)
    
    # 5단계: 텍스트 임베딩 학습 설정
    if self.train_setup_configs["train_text_embedding"]:
        self.model.get_input_embeddings().requires_grad_(True)
    else:
        self.model.get_input_embeddings().requires_grad_(False)
    
    # 6단계: 액션 헤드 학습 설정
    self.act_head.requires_grad_(True)
```

**출처**: `RoboVLMs/robovlms/model/backbone/roboflamingo.py:131-156`

### 7.2 추론 변수 (Inference Variables)

**BaseRoboVLM.inference() - 추론 모드 설정**
```python
def inference(
    self,
    vision_x: torch.Tensor,
    lang_x: torch.Tensor,
    attention_mask: torch.Tensor = None,
    position_ids: torch.LongTensor = None,
    use_cached_vision_x: bool = False,
    action_labels: Tuple[torch.Tensor, torch.Tensor] = None,
    action_mask: torch.Tensor = None,
    caption_labels: torch.Tensor = None,
    caption_mask: torch.Tensor = None,
    past_key_values=None,
    use_cache: bool = False,
    vision_gripper=None,
    **kwargs,
):
    prediction = {}
    
    # 1단계: 입력 검증
    assert vision_x is not None
    bs, seq_len = vision_x.shape[:2]
    action_space = self.act_head_configs.get("action_space", "continuous")
    
    # 2단계: 액션 예측
    if self.train_setup_configs["predict_action"]:
        if action_space == "discrete":
            action = self.pred_action_discrete(
                lang_x, vision_x, vision_gripper, attention_mask
            )
            prediction["action"] = action
        else:
            prediction["action"] = self.forward_continuous(
                vision_x,
                lang_x,
                attention_mask,
                vision_gripper=vision_gripper,
                mode="inference",
            )
    
    return prediction
```

**출처**: `RoboVLMs/robovlms/model/backbone/base_backbone.py:1454-1491`

**BaseModelInference.__init__() - 추론 모델 초기화**
```python
def __init__(
    self,
    ckpt_path,
    configs,
    device,
    save_dir=None,
    unnorm_key: Optional[str] = None,
    policy_setup: str = "widowx_bridge",
    exec_horizon=1,
):
    self.configs = configs
    self.dataset_stat = self.load_dataset_stat()
    self.model = BaseTrainer(configs=configs)
    self.policy = self.model
    
    # 1단계: 환경 변수 설정
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # 2단계: 정책 설정
    if policy_setup == "widowx_bridge":
        unnorm_key = "bridge_orig" if unnorm_key is None else unnorm_key
    elif policy_setup == "google_robot":
        unnorm_key = "fractal20220817_data" if unnorm_key is None else unnorm_key
    
    # 3단계: 그리퍼 액션 설정
    self.sticky_gripper_num_repeat = 2
    self.policy_setup = policy_setup
    self.unnorm_key = unnorm_key
    
    if self.policy_setup == "google_robot":
        self.close_gripper_act = -1
    elif self.policy_setup == "widowx_bridge":
        self.close_gripper_act = 1
    
    # 4단계: 이미지 및 액션 설정
    self.image_size = self.configs.get("image_size", 224)
    self.action_scale = self.configs.get("action_scale", 1.0)
    self.horizon = self.configs["window_size"]
    self.window_size = self.horizon
    self.pred_action_horizon = exec_horizon
```

**출처**: `RoboVLMs/eval/simpler/model_wrapper.py:15-58`

**StandaloneVLAInference.load_model() - 추론 모델 로드**
```python
def load_model(self):
    """VLA 모델 로드"""
    try:
        print(f"📥 모델 로딩 중: {self.model_id}")
        
        model_save_path = Path(self.model_cache_dir) / self.model_id.split('/')[-1]
        model_save_path.mkdir(parents=True, exist_ok=True)

        # 1단계: 프로세서 로드
        self.processor = AutoProcessor.from_pretrained(
            self.model_id, 
            cache_dir=model_save_path
        )

        # 2단계: 모델 로드
        model_kwargs = {
            "cache_dir": model_save_path,
            "low_cpu_mem_usage": True
        }
        
        if self.device.type == "cuda":
            model_kwargs["torch_dtype"] = torch.bfloat16
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["torch_dtype"] = torch.float32

        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            self.model_id, 
            **model_kwargs
        )
        
        if self.device.type != "cuda":
            self.model.to(self.device)
        
        # 3단계: 추론 모드 설정
        self.model.eval()
        print("✅ 모델 로딩 완료")
        
    except Exception as e:
        print(f"❌ 모델 로딩 실패: {e}")
        raise
```

**출처**: `RoboVLMs/vla_test/standalone_vla_test.py:46-85`

### 7.3 학습 vs 추론 변수 비교

| 구분 | 학습 변수 | 추론 변수 |
|------|-----------|-----------|
| **모드** | `model.train()` | `model.eval()` |
| **그래디언트** | `requires_grad=True` | `requires_grad=False` |
| **캐시** | `use_cache=False` | `use_cache=True` |
| **드롭아웃** | 활성화 | 비활성화 |
| **배치 정규화** | 학습 모드 | 평가 모드 |
| **메모리** | 높음 (그래디언트) | 낮음 (그래디언트 없음) |
| **입력** | `action_labels` 포함 | `action_labels` 없음 |
| **출력** | Loss 계산 | 액션 예측만 |
| **토큰 삽입** | Teacher Forcing | Autoregressive |

### 7.4 환경 변수 설정

**Docker 환경 변수 (docker-compose.yml)**
```yaml
environment:
  - DISPLAY=${DISPLAY:-:0}
  - ROS_DOMAIN_ID=42
  - CUDA_VISIBLE_DEVICES=0
  - TORCH_DTYPE=bfloat16
  - PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
  - TRANSFORMERS_CACHE=/workspace/.vlms
  - HF_HOME=/workspace/.vlms
  - PYTHONPATH=/workspace:/workspace/robovlms
  - VLA_MODEL=paligemma-3b-mix-224
  - ACTION_MODE=automotive
  - ACTION_DIM=4
  - WINDOW_SIZE=8
  - INFERENCE_LATENCY_TARGET=100
  - PROJECT_NAME=k_project_event_vla
```

**출처**: `RoboVLMs/docker-compose.yml:25-39`

### 7.5 학습 변수 상세 설정

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

## 🤖 8. 실제 로봇 벤치마크 데이터셋 상세 분석

### 8.1 Real-World Experiments 벤치마크

**벤치마크 개요**
- **총 작업 수**: 105개의 조작 작업
- **데이터 규모**: 70,000개 이상의 원격 조작 인간 궤적
- **평가 설정**: 1개 단순 설정 + 4개 도전적 미지 설정
- **총 평가 작업**: 20개 작업
- **롤아웃**: 각 설정당 3회 롤아웃 (작업당 5개 설정)

**출처**: RoboVLMs 논문 Appendix K, Appendix D, Figure 15-17

**로봇 사양**
- **자유도**: 7-DOF (6차원 자세 + 1차원 그리퍼)
- **관측 정보**: 고유 감각 정보 + 시각 관측 + 언어 입력

### 8.2 CALVIN 벤치마크 상세

**CALVIN [32] - Simulated Benchmark**

**데이터셋 구성**
```python
# CALVIN 데이터셋 구조
calvin_dataset = {
    "demonstrations": 24000,                    # 24k 인간 원격 조작 데모
    "trajectory_length": "< 64 timesteps",      # 각 궤적 64 타임스텝 이하
    "language_annotations": True,               # 언어 명령 포함
    "basic_skills": 34,                         # 34개 사전 정의 기본 스킬
    "splits": ["scene_A", "scene_B", "scene_C", "scene_D"]
}
```

**34개 기본 스킬 목록**
1. rotate blue block right
2. move slider right
3. lift red block slider
4. place slider
5. turn off light bulb
6. turn off led light
7. push in drawer
8. lift blue block drawer
9. close drawer
10. lift pink block slider
11. lift pink block table
12. move slider left
13. open drawer
14. turn on light bulb
15. rotate blue block left
16. push blue block left
17. rotate red block right
18. turn on led light
19. push pink block right
20. push red block left
21. lift blue block table
22. place in drawer
23. rotate red block left
24. push pink block left
25. lift stacked blocks
26. lift blue block slider
27. push red block right

**평가 메트릭**
- **Sequential Task Success Rate**: 5개 연속 작업 완료 성공률
- **Average Length**: 달성한 작업의 평균 길이
- **평가 규모**: D split에서 1000 롤아웃, 각 롤아웃당 5개 연속 서브태스크

**출처**: CALVIN 논문 [32], RoboVLMs 논문

### 8.3 SimplerEnv 벤치마크

**SimplerEnv [25] - Real-to-Sim Evaluation**

**벤치마크 목적**
- 실제 로봇 정책을 시뮬레이션에서 평가
- Google Robot, BridgeData V2와 비교 가능한 아레나 제공
- 효율적이고 확장 가능한 실제 세계 평가 대안

#### 8.3.1 Google Robot 설정 작업

**1) pick coke can**
```python
# pick coke can 작업 설정
task_config = {
    "objective": "빈 코크 캔을 테이블에서 집어 들기",
    "positions": ["horizontal", "vertical", "upright"],  # 3가지 위치
    "grid_points": 25,                                   # 직사각형 영역 내 25개 그리드
    "total_trials": 75,                                  # 25 × 3 = 75 시험
    "distractors": False                                 # 표준 설정에서는 방해 요소 없음
}
```

**2) move {obj1} near {obj2}**
```python
# move near 작업 설정
task_config = {
    "objective": "obj1을 obj2 근처로 이동",
    "objects": ["blue plastic bottle", "Pepsi can", "orange", 
                "7up can", "apple", "sponge", "Coke can", "Redbull can"],  # 8개 물체
    "formation": "triangular",                           # 삼각형 배치
    "triplets": 5,                                       # 5개 triplet (랜덤 선택)
    "patterns": ["upright", "inverted"],                 # 2가지 삼각형 패턴
    "trials_per_triplet": 6,                            # triplet당 6회 시험
    "total_trials": 60                                   # 5 × 6 × 2 = 60 시험
}
```

**3) (open/close) (top/middle/bottom) drawer**
```python
# drawer 작업 설정
task_config = {
    "objective": "특정 서랍 열기/닫기",
    "drawers": 3,                                        # top, middle, bottom
    "actions": ["open", "close"],                        # 2가지 액션
    "robot_positions": 9,                                # 9개 그리드 위치
    "total_trials": 54,                                  # 3 × 2 × 9 = 54 시험
    "evaluation_type": "articulated_objects"             # 관절 물체 처리 능력 평가
}
```

**4) open top drawer; place apple into top drawer**
```python
# multi-step 작업 설정
task_config = {
    "objective": "서랍 열고 사과를 서랍에 넣기",
    "steps": [
        "open top drawer",
        "place apple into top drawer"
    ],
    "robot_positions": 3,                                # 로봇 위치 3개
    "apple_positions": 9,                                # 사과 그리드 위치 9개
    "total_trials": 27,                                  # 3 × 9 = 27 시험
    "instruction_switch": "midpoint or terminate token", # 명령 전환 시점
    "evaluation_type": "sequential_multi-action"         # 순차적 다중 액션 평가
}
```

#### 8.3.2 WidowX + Bridge 설정 작업

**1) put the spoon on the towel**
```python
# spoon on towel 작업 설정
task_config = {
    "objective": "수저를 타월 위에 놓기",
    "square_size": "15 cm",                              # 정사각형 크기
    "spoon_positions": ["corner_1", "corner_2", "corner_3", "corner_4"],
    "towel_positions": ["corner_1", "corner_2", "corner_3", "corner_4"],
    "spoon_orientations": ["horizontal", "vertical"],    # 2가지 방향
    "total_trials": 24,                                  # 4 × 4 × 2 / 2 = 24 시험
    "gripper_adjustment": True                           # 그리퍼 방향 조정 필요
}
```

**2) put carrot on plate**
```python
# carrot on plate 작업 설정
task_config = {
    "objective": "당근을 접시 위에 놓기",
    "square_size": "15 cm",
    "carrot_positions": ["corner_1", "corner_2", "corner_3", "corner_4"],
    "plate_positions": ["corner_1", "corner_2", "corner_3", "corner_4"],
    "total_trials": 24,
    "similar_to": "put the spoon on the towel"
}
```

**3) stack the green block on the yellow block**
```python
# block stacking 작업 설정
task_config = {
    "objective": "초록 블록을 노란 블록 위에 쌓기",
    "block_size": "3 cm",                                # 블록 크기
    "square_configs": [
        {"size": "10 cm", "trials": 12},                 # 10cm 정사각형
        {"size": "20 cm", "trials": 12}                  # 20cm 정사각형
    ],
    "green_block_positions": 4,                          # 4개 코너
    "yellow_block_positions": 4,                         # 4개 코너
    "total_trials": 24                                   # (4 × 4 / 2) × 2 = 24 시험
}
```

**4) put eggplant into yellow basket**
```python
# eggplant into basket 작업 설정
task_config = {
    "objective": "가지를 노란 바구니에 넣기",
    "environment": "sink with two basins",               # 2개 세면대
    "eggplant_location": "right basin (random)",         # 오른쪽 세면대 (랜덤 위치)
    "basket_location": "left basin",                     # 왼쪽 세면대
    "eggplant_variations": {
        "position": "random",
        "orientation": "random",
        "constraint": "easily graspable, away from edges"
    },
    "total_trials": 24
}
```

### 8.4 벤치마크 평가 요약

| 벤치마크 | 유형 | 작업 수 | 데이터 규모 | 평가 메트릭 |
|---------|------|---------|-------------|-------------|
| **Real-World Experiments** | 실제 로봇 | 20개 (105개 중) | 70,000+ 궤적 | 설정별 평균 성공률 |
| **CALVIN** | 시뮬레이션 | 34개 기본 스킬 | 24,000 데모 | Sequential Success Rate, Avg Length |
| **SimplerEnv (Google)** | Real-to-Sim | 4개 작업 | - | 시험별 성공률 (75-54회) |
| **SimplerEnv (Bridge)** | Real-to-Sim | 4개 작업 | - | 시험별 성공률 (24회) |

### 8.5 코드 구현 예시

**DiskCalvinDataset - CALVIN 데이터 로딩**
```python
class DiskCalvinDataset(BaseCalvinDataset):
    """디스크에서 개별 파일로 에피소드를 로드하는 데이터셋"""
    def __init__(
        self,
        image_fn: Callable,
        tokenizer: Callable,
        *args: Any,
        skip_frames: int = 1,
        seq_len: int = 1,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        # ... (초기화 코드)
```

**출처**: `RoboVLMs/robovlms/data/calvin_dataset.py:428-447`

**SimplerEnv 평가 함수**
```python
def evaluate_simpler_env(model, env, task_config):
    """SimplerEnv에서 모델 평가"""
    success_count = 0
    total_trials = task_config["total_trials"]
    
    for trial in range(total_trials):
        # 환경 초기화
        obs = env.reset()
        
        # 모델 추론
        action = model.inference(
            vision_x=obs["rgb"],
            lang_x=task_config["instruction"]
        )
        
        # 액션 실행 및 평가
        success = env.step(action)
        success_count += int(success)
    
    success_rate = success_count / total_trials
    return success_rate
```

## 🎯 9. 핵심 학습 아이디어

### 9.1 VLM의 역할

**1) 멀티모달 이해**
- 이미지와 텍스트를 동시에 이해
- 로봇 환경의 시각적 상황 파악
- 언어 명령의 의미 해석

**2) 특징 추출**
- 이미지에서 로봇 상태 정보 추출
- 텍스트에서 액션 의도 파악
- 멀티모달 융합 특징 생성

### 5.2 LSTM의 역할

**1) 시퀀스 모델링**
- 시간적 의존성 학습
- 이전 액션에 기반한 다음 액션 예측
- 로봇 궤적의 연속성 보장

**2) 액션 예측**
- VLM 특징을 액션으로 변환
- 7-DOF 로봇 액션 생성
- 팔과 그리퍼의 조화로운 제어

### 5.3 학습 전략

**1) End-to-End 학습**
- VLM과 LSTM을 동시에 학습
- 전체 파이프라인의 최적화
- 멀티모달 이해와 액션 예측의 통합

**2) 멀티태스크 학습**
- 액션 예측 (주 태스크)
- 캡션 생성 (보조 태스크)
- 미래 예측 (선택적)

## 📊 6. 학습 효과 분석

### 6.1 VLM Fine-tuning 효과

**Before Fine-tuning**
- 일반적인 이미지-텍스트 이해
- 로봇 환경에 특화되지 않음
- 액션 예측 능력 부족

**After Fine-tuning**
- 로봇 환경에 특화된 이해
- 액션-이미지-텍스트 연관성 학습
- 멀티모달 융합 능력 향상

### 6.2 LSTM 학습 효과

**Before LSTM 학습**
- 단순한 액션 매핑
- 시간적 의존성 부족
- 궤적 연속성 문제

**After LSTM 학습**
- 시퀀스 기반 액션 예측
- 시간적 의존성 학습
- 부드러운 로봇 궤적 생성

## 📊 8. RoboVLMs 논문 데이터 수집 정보

### 8.1 논문 정보

**논문 제목**: "Towards Generalist Robot Policies: What Matters in Building Vision-Language-Action Models"
**발표일**: 2024년 12월 18일
**arXiv 링크**: https://arxiv.org/abs/2412.14058
**저자**: Xinghang Li, Peiyan Li, Minghuan Liu, Dong Wang, Jirong Liu, Bingyi Kang, Xiao Ma, Tao Kong, Hanbo Zhang, Huaping Liu

### 8.2 CALVIN 데이터셋 수집 정보

**CALVIN 논문 정보**
- **논문 제목**: "CALVIN: A Benchmark for Language-Conditioned Policy Learning for Long-Horizon Robot Manipulation Tasks"
- **발표 연도**: 2022년
- **IEEE 논문 링크**: https://ieeexplore.ieee.org/document/9788026
- **GitHub 저장소**: https://github.com/mees/calvin

**데이터 수집 규모**
- **총 데모 수**: 25,000개 demonstrations
- **태스크 수**: 34개 기본 태스크
- **로봇 하드웨어**: Franka Emika Panda 7-DOF 로봇팔
- **데이터 수집 환경**: 실제 물리 환경 (테이블, 물체, 조작 공간)

**출처**: 
- arXiv 논문: https://arxiv.org/abs/2112.03227 (2021년 12월)
- GitHub 저장소: https://github.com/mees/calvin
- Hugging Face 데이터셋: https://huggingface.co/datasets/nhop/calvin

**데이터 수집 방법**
- **전문가 조작**: 숙련된 조작자가 직접 로봇을 제어
- **다중 카메라**: 정적 카메라 + 그리퍼 카메라 시스템
- **언어 주석**: 각 데모에 대한 자연어 설명 추가
- **다양한 태스크**: pick-and-place, navigation, manipulation 등

**출처**: 
- arXiv 논문: https://arxiv.org/abs/2112.03227
- GitHub 저장소: https://github.com/mees/calvin
- Hugging Face 데이터셋: https://huggingface.co/datasets/nhop/calvin

**데이터 구성**
- **이미지 데이터**: RGB 이미지 (224x224)
- **액션 데이터**: 7-DOF 연속 액션 (팔 6-DOF + 그리퍼 1-DOF)
- **언어 데이터**: 자연어 명령 및 설명
- **상태 데이터**: 로봇 관절 상태 및 센서 정보

**CALVIN 데이터셋의 특징**
- **장기적 작업 시퀀스**: "서랍을 열어라", "파란 블록을 서랍에 밀어 넣어라", "서랍을 닫아라"와 같은 일련의 언어 지시
- **연속적 제어**: 30Hz로 연속적인 동작 수행
- **유연한 센서 구성**: 다양한 센서 입력 실험 지원
- **오픈 소스**: 연구자들이 자유롭게 사용하고 확장 가능

### 8.3 데이터 수집의 특징

**1) Real-World 특성**
- 실제 물리 환경에서 수집
- 물리 법칙을 따르는 로봇 동작
- 실제 물체와의 상호작용

**2) 다양성**
- 34개 서로 다른 태스크
- 다양한 물체와 환경
- 다양한 조작 패턴

**3) 품질**
- 전문가 수준의 조작
- 일관된 데이터 품질
- 정확한 언어 주석

### 8.4 논문의 실험 규모

**실험 규모**
- **VLM 백본**: 8개 이상의 다양한 VLM 모델
- **정책 아키텍처**: 4개의 서로 다른 아키텍처
- **총 실험 수**: 600개 이상의 실험
- **평가 환경**: 시뮬레이션 + 실제 환경

**실험 범위**
- 다양한 VLM 백본 비교
- 정책 아키텍처 비교
- 데이터 분포 영향 분석
- 학습 방법 비교

## 🎯 9. 핵심 요약

### 9.1 전체 파이프라인

1. **Real-World 데이터 수집**: CALVIN 데이터셋 (25k demonstrations, 34 tasks)
2. **VLM Fine-tuning**: 멀티모달 이해 능력 향상
3. **LSTM 학습**: 시퀀스 기반 액션 예측
4. **End-to-End 최적화**: 전체 파이프라인 통합 학습

### 9.2 핵심 아이디어

**VLM의 역할**: 멀티모달 이해 + 특징 추출
**LSTM의 역할**: 시퀀스 모델링 + 액션 예측
**학습 전략**: End-to-End + 멀티태스크 학습

### 9.3 학습 순서

1. **데이터 전처리**: 이미지 정규화 + 액션 정규화
2. **VLM Fine-tuning**: 멀티모달 이해 능력 학습
3. **LSTM 학습**: 시퀀스 기반 액션 예측 학습
4. **통합 최적화**: 전체 파이프라인 End-to-End 학습

### 9.4 논문 참고 정보

**논문 링크**: https://arxiv.org/abs/2412.14058
**공식 웹사이트**: https://robovlms.github.io/
**GitHub 저장소**: RoboVLMs 프로젝트

이 분석을 통해 RoboVLMs의 VLM Fine-tuning과 LSTM Layer 학습 과정을 명확히 이해할 수 있습니다.
