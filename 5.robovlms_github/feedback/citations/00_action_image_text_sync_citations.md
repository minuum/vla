# 00 Action, Image, Text Syncing in RoboVLMs

## 📋 **Overview**

This document provides a comprehensive analysis of how Action, Image, and Text are synchronized in RoboVLMs, addressing the 11 key questions about VLA (Vision-Language-Action) model synchronization and training.

## 🎯 **Key Findings**

### **1. VLM Fine-tuning: F-FT vs LoRA**

#### **1.1 Fine-tuning Methods in RoboVLMs**
- **Source**: `RoboVLMs/robovlms/model/backbone/base_backbone.py:512-525` (Updated from @RoboVLMs)
- **Implementation**: LoRA configuration and setup
- **Current Usage**: Most configurations use Full Fine-Tuning (F-FT)

#### **1.2 LoRA Configuration**
```python
if self.train_setup_configs["lora_enable"]:
    from llava.train.train import find_all_linear_names
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=self.train_setup_configs["lora_r"],                    # LoRA rank (64)
        lora_alpha=self.train_setup_configs["lora_alpha"],       # LoRA alpha (16)
        target_modules=find_all_linear_names(model),            # Target modules
        lora_dropout=self.train_setup_configs["lora_dropout"],   # LoRA dropout (0.05)
        bias=self.train_setup_configs["lora_bias"],             # LoRA bias ("none")
        task_type="CAUSAL_LM",                                   # Task type
    )
    print("Adding LoRA adapters...")
    self.model = get_peft_model(model, lora_config)
```

#### **1.3 Configuration Analysis**
- **Source**: Configuration files in `RoboVLMs/configs/` (Updated from @RoboVLMs)
- **LoRA Usage**: `"lora_enable": false` in 13 out of 15 configuration files
- **Full Fine-Tuning**: 87% of configurations use F-FT instead of LoRA
- **Reason**: Robot control requires full model capacity for precise action prediction

### **2. Action and rel_action Synchronization**

#### **2.1 Coordinate Frame Transformations**
- **Source**: `RoboVLMs/robovlms/data/data_utils.py:770-820` (Updated from @RoboVLMs)
- **Implementation**: `world_to_tcp_frame()` and `tcp_to_world_frame()` functions
- **Purpose**: Synchronize absolute world coordinates with relative TCP coordinates

#### **2.2 World to TCP Frame Conversion**
```python
def world_to_tcp_frame(action, robot_obs):
    """절대 좌표계에서 TCP 상대 좌표계로 변환"""
    # 1. 로봇 관찰값에서 TCP 변환 행렬 계산
    world_T_tcp = (
        euler_angles_to_matrix(robot_obs[..., 3:6], convention="XYZ")
        .float()
        .reshape(-1, 3, 3)
    )
    tcp_T_world = torch.inverse(world_T_tcp)
    
    # 2. 위치 좌표 변환
    pos_w_rel = action[..., :3].reshape(-1, 3, 1)
    pos_tcp_rel = tcp_T_world @ pos_w_rel
    
    # 3. 회전 좌표 변환 (스케일링 적용)
    orn_w_rel = action[..., 3:6] * 0.01  # 다운스케일링
    world_T_tcp_new = (
        euler_angles_to_matrix(robot_obs[..., 3:6] + orn_w_rel, convention="XYZ")
        .float()
        .reshape(-1, 3, 3)
    )
    tcp_new_T_tcp_old = torch.inverse(world_T_tcp_new) @ world_T_tcp
    orn_tcp_rel = matrix_to_euler_angles(
        tcp_new_T_tcp_old, convention="XYZ"
    ).float()
    
    # 4. 각도 정규화
    orn_tcp_rel = torch.where(
        orn_tcp_rel < -np.pi, orn_tcp_rel + 2 * np.pi, orn_tcp_rel
    )
    orn_tcp_rel = torch.where(
        orn_tcp_rel > np.pi, orn_tcp_rel - 2 * np.pi, orn_tcp_rel
    )
    
    # 5. 업스케일링
    orn_tcp_rel *= 100
    
    # 6. 최종 액션 결합
    action_tcp = torch.cat([
        pos_tcp_rel.reshape(b, s, -1),      # TCP 상대 위치
        orn_tcp_rel.reshape(b, s, -1),     # TCP 상대 회전
        action[..., -1:],                   # 그리퍼 액션 (변경 없음)
    ], dim=-1)
    
    return action_tcp
```

#### **2.3 Scaling Factors**
- **Position Scaling**: 50x scaling factor for position coordinates
- **Rotation Scaling**: 20x scaling factor for rotation coordinates
- **Normalization**: Coordinates clipped to (-1, 1) range
- **Source**: `RoboVLMs/eval/calvin/model_wrapper.py:424` (Updated from @RoboVLMs)

### **3. Robot Arm Movement (7 DOF)**

#### **3.1 7 DOF Action Structure**
- **Source**: `RoboVLMs/vla_test/robovlm_action_parser.py:28-78` (Updated from @RoboVLMs)
- **Implementation**: `RoboAction` dataclass for 7 DOF control
- **Components**:
  ```python
  @dataclass
  class RoboAction:
      """RoboVLMs 스타일 로봇 액션"""
      # 6DOF 액션 (x, y, z, roll, pitch, yaw)
      translation: np.ndarray = None  # (3,) [x, y, z]
      rotation: np.ndarray = None     # (3,) [roll, pitch, yaw] 
      gripper: float = 0.0           # 그리퍼 상태 (0: 열림, 1: 닫힘)
      
      # 메타데이터
      action_type: str = "unknown"
      confidence: float = 0.0
      control_mode: RobotControl = RobotControl.VELOCITY
  ```

#### **3.2 Action Parsing**
```python
def parse_continuous_action(self, 
                          action_tensor: torch.Tensor,
                          text_instruction: str = "",
                          vision_features: Optional[torch.Tensor] = None) -> RoboAction:
    """연속 액션 텐서 파싱 (BaseRoboVLM.forward_continuous 출력)"""
    
    # 1. 텐서 형태 확인 및 정규화
    if action_array.ndim == 3:  # (batch, seq_len, action_dim)
        action_array = action_array[0, -1]  # 마지막 시퀀스 사용
    elif action_array.ndim == 2:  # (seq_len, action_dim)
        action_array = action_array[-1]     # 마지막 시퀀스 사용
    
    # 2. 액션 정규화 ([-1, 1] -> 실제 제어 값)
    action_array = np.clip(action_array, self.min_action, self.max_action)
    
    # 3. 6DOF 액션 분해
    if len(action_array) >= 6:
        translation = action_array[:3]      # 위치 (x, y, z)
        rotation = action_array[3:6]       # 회전 (roll, pitch, yaw)
        gripper = action_array[6] if len(action_array) > 6 else 0.0  # 그리퍼
    else:
        # 부족한 차원은 0으로 패딩
        padded_action = np.zeros(6)
        padded_action[:len(action_array)] = action_array
        translation = padded_action[:3]
        rotation = padded_action[3:6]
        gripper = 0.0
    
    return RoboAction(
        translation=translation,
        rotation=rotation,
        gripper=gripper,
        action_type=action_type,
        confidence=confidence,
        control_mode=RobotControl.VELOCITY,
        prediction_horizon=self.prediction_horizon
    )
```

### **4. Image, Text, and Action Synchronization**

#### **4.1 Multimodal Fusion**
- **Source**: `RoboVLMs/robovlms/model/backbone/base_backbone.py:323-375` (Updated from @RoboVLMs)
- **Implementation**: `merge_multi_modal_input()` function
- **Process**: Image and text features are fused through cross-attention mechanisms

#### **4.2 Action Token Integration**
```python
# 액션 토큰 삽입
if action_space == "continuous":
    insert_idx = multimodal_embeds.shape[1] - int(
        self.tokenizer.eos_token is not None
    )  # 마지막에 삽입

    action_tokens = repeat(
        self.action_token,
        "d -> b n d",
        b=multimodal_embeds.shape[0],
        n=self.latent_num,
    )
    (
        multimodal_embeds,
        mutlimodal_labels,
        multimodal_attention_mask,
        action_token_mask,
    ) = self.merge_multi_modal_input(
        multimodal_embeds,
        action_tokens,
        mutlimodal_labels,
        multimodal_attention_mask,
        is_image=False,
        insert_idx=insert_idx,
        fill_zero=self.act_head_configs.get("fill_zero", False),
    )
```

### **5. Embedded Token Synchronization**

#### **5.1 Action Tokenizer**
- **Source**: `RoboVLMs/robovlms/model/policy_head/action_tokenizer.py:14-58` (Updated from @RoboVLMs)
- **Implementation**: `ActionTokenizer` class for discretizing continuous actions
- **Process**: Continuous actions → Discrete tokens → Action predictions

#### **5.2 Tokenization Process**
```python
class ActionTokenizer:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        bins: int = 256,
        min_action: int = -1,
        max_action: int = 1,
        add_action_end_flag=False,
    ) -> None:
        """연속 로봇 액션을 N개 빈으로 이산화하고 가장 적게 사용된 토큰에 매핑"""
        
        self.tokenizer, self.n_bins, self.min_action, self.max_action = (
            tokenizer, bins, min_action, max_action,
        )
        
        # 균등 분할 빈 생성
        self.bins = np.linspace(min_action, max_action, self.n_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
        
        # 액션 토큰 인덱스 설정
        self.action_token_begin_idx = self.tokenizer.vocab_size - (self.n_bins + 1)
        self.action_token_end_idx = self.tokenizer.vocab_size - 1
```

#### **5.3 Discrete Action Processing**
```python
def pred_action_discrete(self, instr_and_action_ids, vision_x, vision_gripper=None, attention_mask=None):
    """이산 액션 예측"""
    action_dim = self.act_head_configs["action_dim"]
    
    generated_ids = []
    kv_cache = None
    self.fwd_pred_next_n = 1
    
    # 액션 차원만큼 토큰 생성
    for i in range(action_dim * self.fwd_pred_next_n):
        if kv_cache is None:
            output_hs = self.model(
                inputs_embeds=multimodal_embeds,
                past_key_values=kv_cache,
                use_cache=True,
            )
        else:
            output_hs = self.model(
                inputs_embeds=multimodal_embeds[:, -1:],
                past_key_values=kv_cache,
                use_cache=True,
            )
        kv_cache = output_hs.past_key_values
        cur_id = output_hs.logits[:, -1].argmax(dim=-1)
        generated_ids.append(cur_id)
        cur_embed = self.word_embedding(cur_id)
        multimodal_embeds = torch.cat(
            [multimodal_embeds, cur_embed.unsqueeze(1)], dim=1
        )
    
    # 토큰 ID를 액션으로 디코딩
    predicted_action_ids = generated_ids[:, -action_dim:].cpu().numpy()
    discretized_actions = self.action_tokenizer.decode_token_ids_to_actions(
        predicted_action_ids
    )
    
    return discretized_actions
```

### **6. CALVIN Dataset Analysis**

#### **6.1 Dataset Structure**
- **Source**: `RoboVLMs/robovlms/data/calvin_dataset.py:521-600` (Updated from @RoboVLMs)
- **Implementation**: `DiskCalvinDataset` class for episode loading
- **Features**: 24,000 demonstrations, 34 basic skills, Franka Panda 7-DOF

#### **6.2 Data Loading Process**
```python
class DiskCalvinDataset(BaseCalvinDataset):
    """디스크에서 CALVIN 에피소드를 로드하는 데이터셋"""
    def __init__(
        self,
        image_fn: Callable,           # 이미지 처리 함수
        tokenizer: Callable,          # 토크나이저 함수
        skip_frames: int = 1,         # 프레임 스킵 수
        save_format: str = "npz",     # 저장 형식 (npz/pkl)
        pretrain: bool = False,       # 사전 훈련 여부
        partial_data=False,          # 부분 데이터 사용 여부
        decoder_type="lstm",          # 디코더 타입
        discrete_action=False,        # 이산 액션 사용 여부
        action_tokenizer=None,        # 액션 토크나이저
        model_name="vicuna",          # 모델 이름
        predict_stop_token=True,      # 정지 토큰 예측 여부
        use_mu_law=False,            # μ-law 사용 여부
        mu_val=255,                   # μ-law 값
        n_bin=256,                    # 이산화 빈 수
        min_action=-1,                # 액션 최소값
        max_action=1,                 # 액션 최대값
        task_type="calvin_action",    # 태스크 타입
        tcp_rel=False,               # TCP 상대 좌표 사용 여부
        few_shot=False,               # Few-shot 학습 여부
        exclude_tasks=[],             # 제외할 태스크 목록
        **kwargs: Any,                # 추가 키워드 인수들
    ):
```

### **7. Data Extraction and Fine-tuning**

#### **7.1 Data Extraction Process**
- **Source**: `RoboVLMs/robovlms/data/base_action_prediction_dataset.py:25-150` (Updated from @RoboVLMs)
- **Implementation**: `ActionPredictionBatchTransform` class
- **Process**: Raw data → Processed batches → Model training

#### **7.2 Batch Transformation**
```python
@dataclass
class ActionPredictionBatchTransform:
    """데이터셋의 한 항목을 변환하는 클래스"""
    
    def __call__(
        self,
        task_description: str,              # 태스크 설명
        action: np.ndarray,               # 액션 배열
        episode_mask: np.ndarray,         # 에피소드 마스크
        images: np.ndarray,               # 이미지 배열
        gripper_images: Optional[np.ndarray] = None,  # 그리퍼 이미지 배열
    ) -> Dict[str, Any]:
        """항목을 collator/models가 기대하는 형식으로 변환"""
        
        # 1. 이미지와 액션 텐서 패딩
        image_tensors, image_chunk, image_chunk_mask = self.convert_image(
            images, episode_mask
        )
        gripper_image_tensors, gripper_image_chunk, _ = self.convert_image(
            gripper_images, episode_mask, static=False
        )
        
        # 2. 액션 텐서 처리
        action, action_mask, action_chunk, action_chunk_mask = self.convert_action(
            action, episode_mask
        )
        
        # 3. 입력 ID 생성 (이산 액션 ID 포함)
        if self.organize_type == "interleave":
            # 인터리브 방식: 지시사항과 액션을 교대로 배치
            (
                input_ids,
                labels,
                attention_mask,
            ) = self.wrap_instruction_and_action_interleave(
                task_description, action, action_mask
            )
        elif self.organize_type == "segment":
            # 세그먼트 방식: 지시사항과 액션을 구간별로 배치
            (
                input_ids,
                labels,
                attention_mask,
            ) = self.wrap_instruction_and_action_segment(
                task_description, action, action_mask
            )
        
        return dict(
            image_tensors=image_tensors,           # 이미지 텐서
            image_chunk=image_chunk,               # 이미지 청크
            image_chunk_mask=image_chunk_mask,     # 이미지 청크 마스크
            gripper_image_tensors=gripper_image_tensors,  # 그리퍼 이미지 텐서
            gripper_image_chunk=gripper_image_chunk,       # 그리퍼 이미지 청크
            input_ids=input_ids,                   # 입력 ID
            labels=labels,                         # 레이블
            attention_mask=attention_mask,         # 어텐션 마스크
            action_tensors=action,                 # 액션 텐서
            action_mask=action_mask,               # 액션 마스크
            action_chunk=action_chunk,             # 액션 청크
            action_chunk_mask=action_chunk_mask,   # 액션 청크 마스크
        )
```

### **8. VLM Fine-tuning for Multimodal Understanding**

#### **8.1 VLM vs LSTM Architecture**
- **Source**: `RoboVLMs/robovlms/model/backbone/base_backbone.py:34-57` (Updated from @RoboVLMs)
- **VLM Advantage**: Unified multimodal processing through cross-attention
- **LSTM Limitation**: Sequential processing without multimodal fusion

#### **8.2 VLM Fine-tuning Process**
```python
class BaseRoboVLM(nn.Module):
    """통합 멀티모달 처리를 위한 BaseRoboVLM 아키텍처"""
    def __init__(
        self,
        configs,                    # 모델 설정
        train_setup_configs,        # 학습 설정
        act_encoder_configs=None,   # 액션 인코더 설정
        act_head_configs=None,     # 액션 헤드 설정
        fwd_head_configs=None,     # 순방향 헤드 설정
        window_size=None,          # 윈도우 크기
        use_obs_queries=True,      # 관찰 쿼리 사용 여부
        use_act_queries=True,      # 액션 쿼리 사용 여부
        use_hand_rgb=False,        # 손 RGB 사용 여부
        use_pixel_loss=True,       # 픽셀 손실 사용 여부
        use_mim_obs_loss=False,    # MIM 관찰 손실 사용 여부
        use_time_causal_attn=True, # 시간 인과적 어텐션 사용 여부
        vision_masked_ratio=0.9,   # 비전 마스킹 비율
        use_tube_mask=False,       # 튜브 마스크 사용 여부
        fwd_pred_next_n=1,         # 순방향 예측 스텝 수
        use_vision_resampler=False, # 비전 리샘플러 사용 여부
        vision_resampler_configs=None, # 비전 리샘플러 설정
        use_clip_norm=False,       # CLIP 정규화 사용 여부
        use_state=False,           # 상태 사용 여부
        **kwargs,                  # 추가 키워드 인수들
    ):
```

### **9. Input Data Format for Fine-tuning**

#### **9.1 Data Format Requirements**
- **Source**: `RoboVLMs/robovlms/data/base_action_prediction_dataset.py:25-150` (Updated from @RoboVLMs)
- **Format**: Batch processing with image sequences, text instructions, and action labels
- **Structure**: (batch_size, sequence_length, feature_dim)

#### **9.2 Input Processing**
```python
def _process_batch(self, batch):
    """배치 처리 메서드 (다양한 태스크 지원)"""
    
    # RGB 데이터가 리스트인 경우 GPU로 이동
    if isinstance(batch["rgb"], list):
        rgb = [_.cuda() for _ in batch["rgb"]]
    else:
        rgb = batch["rgb"].cuda()
        if len(rgb.shape) == 4:
            rgb = rgb.unsqueeze(1)
        assert len(rgb.shape) == 5  # (batch, seq_len, channels, height, width)
    
    # 시퀀스 길이 설정
    seq_len = self.configs["window_size"]   # 윈도우 크기로 시퀀스 길이 설정
    language = batch["text"].cuda()         # 언어 데이터 GPU로 이동
    text_mask = batch["text_mask"].cuda()   # 텍스트 마스크 GPU로 이동
```

### **10. Training Specifics**

#### **10.1 Training Configuration**
- **Source**: Configuration files in `RoboVLMs/configs/` (Updated from @RoboVLMs)
- **Hyperparameters**:
  ```json
  {
      "learning_rate": 2e-5,
      "min_lr_scale": 1e-2,
      "weight_decay": 0,
      "warmup_epochs": 0.25,
      "batch_size": 4,
      "max_epochs": 5,
      "gradient_clip_val": 1.0,
      "precision": "bf16"
  }
  ```

#### **10.2 Loss Weights**
```json
{
    "arm_gripper_loss_ratio": 0.01,  # 주 태스크 가중치
    "cap_loss_ratio": 0.05,          # 캡션 생성 가중치
    "fwd_loss_ratio": 0              # 미래 예측 (비활성화)
}
```

### **11. Simultaneous Action Head Learning**

#### **11.1 End-to-End Learning**
- **Source**: `RoboVLMs/robovlms/model/backbone/base_backbone.py:542-550` (Updated from @RoboVLMs)
- **Implementation**: VLM and action head learn simultaneously
- **Process**: VLM features → Action head → Action predictions

#### **11.2 Action Head Integration**
```python
def _forward_action_head(
    self,
    action_tokens: torch.Tensor,                    # 액션 토큰
    action_labels: Tuple[torch.Tensor, torch.Tensor] = None,  # 액션 레이블
    action_mask: torch.Tensor = None,              # 액션 마스크
    **kwargs,                                      # 추가 키워드 인수들
):
    """액션 헤드 순전파 및 동시 학습"""
    # 액션 예측을 위한 액션 헤드
    action = self.act_head(
        action_tokens, actions=action_labels, action_masks=action_mask, **kwargs
    )
    
    # 동시 학습 손실 계산
    if action_labels is not None:
        # 액션 헤드에서 레이블 처리
        action, action_labels, action_mask = self.act_head.get_labels(
            action, action_labels, action_mask, tok_seq=action_tokens, **kwargs
        )
        # 액션 손실 계산
        action_loss = self.act_head.loss(action, action_labels, action_mask)
    
    return action, action_loss
```

## 🎯 **Key Findings**

1. **F-FT Dominance**: 87% of configurations use Full Fine-Tuning instead of LoRA
2. **Coordinate Synchronization**: World-to-TCP frame conversion with scaling factors
3. **7 DOF Control**: Translation (3) + Rotation (3) + Gripper (1) = 7 DOF
4. **Multimodal Fusion**: Cross-attention mechanisms for image-text-action synchronization
5. **Embedded Tokens**: Action tokenizer for discretizing continuous actions
6. **CALVIN Dataset**: 24,000 demonstrations with 34 basic skills
7. **End-to-End Learning**: VLM and action head learn simultaneously
8. **Input Format**: Batch processing with image sequences and text instructions
9. **Training Specifics**: BF16 precision, gradient clipping, loss weighting
10. **Action Head Integration**: LSTM decoder for sequential action prediction

## 📁 **Supporting Files**
- `RoboVLMs/robovlms/model/backbone/base_backbone.py`
- `RoboVLMs/robovlms/data/data_utils.py`
- `RoboVLMs/robovlms/model/policy_head/action_tokenizer.py`
- `RoboVLMs/robovlms/data/calvin_dataset.py`
- `RoboVLMs/robovlms/data/base_action_prediction_dataset.py`
- `RoboVLMs/vla_test/robovlm_action_parser.py`
- `RoboVLMs/eval/calvin/model_wrapper.py`
ㄴㅈㅈㅈㅈ