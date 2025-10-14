# 6. Embedded Token Synchronization - Citation Evidence

## 🔍 **GitHub Code Implementation (100% Confirmed from @RoboVLMs)**

### **6.1 Action Tokenizer Implementation**
- **File**: `RoboVLMs/robovlms/model/policy_head/action_tokenizer.py:14-95` (Updated from @RoboVLMs)
- **Implementation**: `ActionTokenizer` class for discrete action tokenization
- **Code**:
```python
class ActionTokenizer:
    """연속 액션을 이산 토큰으로 변환하는 토크나이저"""
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,  # 기본 토크나이저
        bins: int = 256,                     # 이산화 빈 수
        min_action: int = -1,               # 액션 최소값
        max_action: int = 1,                # 액션 최대값
        add_action_end_flag=False,          # 액션 끝 플래그 추가 여부
    ) -> None:
        """
        연속 로봇 액션을 차원당 N개 빈으로 이산화하고 가장 적게 사용된 토큰에 매핑
        """
        # 기본 설정 저장
        self.tokenizer, self.n_bins, self.min_action, self.max_action = (
            tokenizer, bins, min_action, max_action,
        )
        
        # 균등 빈 생성 및 빈 중심값 계산
        self.bins = np.linspace(min_action, max_action, self.n_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
        
        # 액션 토큰 인덱스 설정
        self.action_token_begin_idx: int = int(
            self.tokenizer_orig_size - (self.n_bins + 1)  # 액션 토큰 시작 인덱스
        )
        self.action_token_end_idx = self.tokenizer_orig_size  # 액션 토큰 끝 인덱스
```

### **6.2 Action Token Encoding**
- **File**: `RoboVLMs/robovlms/model/policy_head/action_tokenizer.py:82-95` (Updated from @RoboVLMs)
- **Implementation**: `encode_actions_to_token_ids()` and `decode_token_ids_to_actions()` functions
- **Code**:
```python
def encode_actions_to_token_ids(self, action: np.ndarray) -> np.ndarray:
    """연속 액션을 토큰 ID로 인코딩"""
    # 액션을 지정된 범위로 클리핑
    action = np.clip(
        action, a_min=float(self.min_action), a_max=float(self.max_action)
    )
    # 액션을 빈으로 이산화
    discretized_action = np.digitize(action, self.bins)
    
    # 1차원인 경우 리스트로 변환
    if len(discretized_action.shape) == 1:
        return list(self.tokenizer_orig_size - discretized_action)
    else:
        # 다차원인 경우 배열로 변환
        return np.array(self.tokenizer_orig_size - discretized_action).tolist()

def decode_token_ids_to_actions(self, action_token_ids: np.ndarray) -> np.ndarray:
    """토큰 ID를 연속 액션으로 디코딩"""
    # 토큰 ID를 연속 액션으로 변환하는 구현 세부사항
```

### **6.3 Discrete Action Decoder**
- **File**: `RoboVLMs/robovlms/model/policy_head/base_policy.py:173-227` (Updated from @RoboVLMs)
- **Implementation**: `DiscreteDecoder` class with ActionTokenizer integration
- **Code**:
```python
class DiscreteDecoder(BasePolicyHead):
    """이산 액션 디코더 (ActionTokenizer 통합)"""
    def __init__(
        self,
        hidden_size,              # 히든 상태 크기
        action_dim,               # 액션 차원 (7 DOF)
        action_space="continuous", # 액션 공간 타입
        n_bin=256,                # 이산화 빈 수
        min_action=-1,            # 액션 최소값
        max_action=1,             # 액션 최대값
        tokenizer=None,           # 토크나이저
        **kwargs,
    ):
        super().__init__(hidden_size, action_dim, action_space, **kwargs)
        self.n_bin = n_bin                    # 이산화 빈 수
        self.min_action = min_action          # 액션 최소값
        self.max_action = max_action          # 액션 최대값

        # ActionTokenizer import 및 초기화
        from robovlms.model.policy_head.action_tokenizer import ActionTokenizer

        self.action_tokenizer = ActionTokenizer(
            tokenizer,                    # 토크나이저
            bins=self.n_bin,              # 빈 수
            min_action=self.min_action,   # 최소 액션값
            max_action=self.max_action,   # 최대 액션값
        )
```

### **6.3 Action Token Integration in Multimodal Input**
- **File**: `RoboVLMs/robovlms/model/backbone/base_backbone.py:1097-1121`
- **Implementation**: Action token insertion into multimodal embeddings
- **Code**:
```python
if action_space == "continuous":
    # 연속 액션 공간: EOS 토큰 직전에 action token 삽입
    insert_idx = multimodal_embeds.shape[1] - int(
        self.tokenizer.eos_token is not None  # EOS 토큰 존재 여부에 따른 인덱스 조정
    )
    
    # Learnable action token을 배치 크기만큼 복제
    action_tokens = repeat(
        self.action_token,
        "d -> b n d",
        b=multimodal_embeds.shape[0],
        n=self.latent_num,
    )
    
    # 멀티모달 임베딩에 action token 통합
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

### **6.4 Multimodal Input Merging Function**
- **File**: `RoboVLMs/robovlms/model/backbone/base_backbone.py:323-375`
- **Implementation**: `merge_multi_modal_input` function for token integration
- **Code**:
```python
def merge_multi_modal_input(
    self,
    input_embeds: torch.Tensor,
    multimodal_feats: torch.Tensor = None,
    labels: torch.Tensor = None,
    attention_mask: torch.Tensor = None,
    is_image=True,
    insert_idx=1,
    fill_zero=False,
):
    # Action token의 경우 is_image=False로 처리
    if is_image:
        rgb_feats = self.encode_images(multimodal_feats)
        # 이미지 토큰 처리 로직
    else:
        rgb_feats = multimodal_feats  # Action token 직접 사용
    
    added_seq_len = rgb_feats.shape[1]
    
    # 입력 임베딩에 멀티모달 특징 통합
    multimodal_embeds = torch.cat(
        [input_embeds[:, :insert_idx], rgb_feats, input_embeds[:, insert_idx:]],
        dim=1,
    )
```

### **6.5 Discrete Action Prediction**
- **File**: `RoboVLMs/robovlms/model/backbone/base_backbone.py:1384-1452`
- **Implementation**: Token-based action prediction for discrete actions
- **Code**:
```python
def pred_action_discrete(
    self, instr_and_action_ids, vision_x, vision_gripper=None, attention_mask=None
):
    action_dim = self.act_head_configs["action_dim"]
    generated_ids = []
    kv_cache = None
    
    # 액션 차원 × 미래 예측 스텝만큼 토큰 생성
    for i in range(action_dim * self.fwd_pred_next_n):
        output_hs = self.model(
            inputs_embeds=multimodal_embeds,
            past_key_values=kv_cache,
            use_cache=True,
        )
        kv_cache = output_hs.past_key_values
        cur_id = output_hs.logits[:, -1].argmax(dim=-1)
        generated_ids.append(cur_id)
    
    # 생성된 토큰 ID를 연속 액션으로 디코딩
    predicted_action_ids = generated_ids[:, -action_dim:].cpu().numpy()
    discretized_actions = self.action_tokenizer.decode_token_ids_to_actions(
        predicted_action_ids
    )
    
    # 그리퍼 액션 이진화 처리
    discretized_actions[:, -1] = np.where(discretized_actions[:, -1] > 0, 1, -1)
    
    return discretized_actions
```

### **6.6 Action Token Processing in Data Pipeline**
- **File**: `RoboVLMs/robovlms/data/calvin_dataset.py:750-780`
- **Implementation**: Action token encoding in dataset
- **Code**:
```python
def wrap_instruction_and_action(self, lang, action, action_mask):
    # 액션을 토큰 ID로 인코딩
    action_ids = self.action_tokenizer.encode_actions_to_token_ids(action)
    
    # 대화 형식으로 프롬프트 구성
    conversation = [
        {
            "from": "human",
            "value": f"What action should the robot take to {lang}?",
        },
        {"from": "gpt", "value": ""},
    ]
    
    # 입력 ID와 액션 ID 결합
    input_ids = self.tokenizer(
        prompt_builder.get_prompt(), add_special_tokens=True
    ).input_ids
```

## 📊 **Token Synchronization Evidence**

### **6.7 Continuous vs Discrete Action Token Processing**

#### **Continuous Action Space (연속 액션 공간)**
- **Learnable Token**: `nn.Parameter(torch.zeros(self.hidden_size))` 생성
- **Token Integration**: `merge_multi_modal_input()` 함수로 멀티모달 임베딩에 통합
- **Processing**: 직접적인 연속 값 처리, 토큰화 없음
- **Usage**: 정밀한 로봇 제어, 실시간 제어

#### **Discrete Action Space (이산 액션 공간)**
- **Action Tokenizer**: 256개 빈으로 연속 액션을 이산화
- **Token Encoding**: `encode_actions_to_token_ids()` 함수로 토큰 ID 변환
- **Token Decoding**: `decode_token_ids_to_actions()` 함수로 연속 액션 복원
- **Usage**: 언어 모델 호환성, 시퀀스 모델링

### **6.8 Multimodal Token Fusion Process**

#### **Token Integration Pipeline**
1. **Vision Tokens**: 이미지 특징을 토큰으로 변환
2. **Language Tokens**: 텍스트 명령을 토큰으로 변환
3. **Action Tokens**: 액션을 토큰으로 변환 (연속/이산)
4. **Fusion**: `merge_multi_modal_input()` 함수로 통합

#### **Token Synchronization Mechanism**
```python
# 1. 멀티모달 임베딩 생성
multimodal_embeds = self.merge_multi_modal_input(
    vision_embeds, language_embeds, action_embeds
)

# 2. 통합된 임베딩으로 모델 실행
output = self.model(
    inputs_embeds=multimodal_embeds,
    attention_mask=multimodal_attention_mask
)

# 3. 액션 예측 및 디코딩
predicted_actions = self.decode_action_tokens(output.logits)
```

### **6.9 End-to-End Learning Process**

#### **Training Phase**
- **Joint Optimization**: Vision, Language, Action 토큰 동시 학습
- **Loss Calculation**: 멀티모달 손실 함수로 통합 학습
- **Gradient Flow**: 모든 토큰 타입에 대한 그래디언트 전파

#### **Inference Phase**
- **Token Generation**: 시퀀스 모델로 액션 토큰 생성
- **Action Decoding**: 토큰을 연속 액션으로 변환
- **Robot Control**: 변환된 액션으로 로봇 제어

## 🎯 **Key Findings**

### **6.10 Technical Innovations**

1. **Learnable Action Tokens**: 
   - 연속 액션 공간에서 학습 가능한 액션 토큰 생성
   - `nn.Parameter`로 구현된 학습 가능한 파라미터

2. **Discrete Tokenization**: 
   - 256개 빈으로 연속 액션을 이산화
   - 언어 모델과의 호환성 확보

3. **Multimodal Fusion**: 
   - Vision, Language, Action 토큰의 통합 처리
   - `merge_multi_modal_input()` 함수로 구현

4. **End-to-End Learning**: 
   - 모든 토큰 타입의 동시 최적화
   - 멀티모달 손실 함수 활용

### **6.11 Implementation Details**

#### **Token Creation Process**
```python
# Continuous: Learnable parameter 생성
self.action_token = nn.Parameter(torch.zeros(self.hidden_size))

# Discrete: ActionTokenizer로 토큰화
action_tokenizer = ActionTokenizer(tokenizer, bins=256)
```

#### **Token Integration Process**
```python
# 멀티모달 임베딩에 액션 토큰 통합
multimodal_embeds = self.merge_multi_modal_input(
    input_embeds, action_tokens, labels, attention_mask,
    is_image=False, insert_idx=insert_idx
)
```

#### **Token Processing Process**
```python
# 이산 액션 예측
predicted_actions = self.pred_action_discrete(
    instr_and_action_ids, vision_x, vision_gripper
)

# 연속 액션 예측  
predicted_actions = self.forward_continuous(
    vision_x, lang_x, attention_mask
)
```

## 📁 **Supporting Files**
- `RoboVLMs/robovlms/model/backbone/base_backbone.py` (L124-126, L323-375, L1097-1121, L1384-1452)
- `RoboVLMs/robovlms/model/policy_head/action_tokenizer.py` (L14-58)
- `RoboVLMs/robovlms/data/calvin_dataset.py` (L750-780)
- `RoboVLMs/robovlms/model/policy_head/base_policy.py` (L173-227)
- `RoboVLMs/robovlms/data/base_action_prediction_dataset.py` (L141-226)
