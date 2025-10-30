# VLM + LSTM 통합 아키텍처

## 1. 전체 아키텍처 개요

### 1.1 시스템 구성도

```
Input Data
    ↓
┌─────────────────────────────────────────────────────────┐
│                    Multi-modal Input                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │   Image     │  │    Text     │  │  [LRN]      │    │
│  │ (2 cameras) │  │ (language)  │  │ (learnable) │    │
│  └─────────────┘  └─────────────┘  └─────────────┘    │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│                  VLM Backbone                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │   Vision    │  │    Text     │  │  Attention  │    │
│  │  Encoder    │  │  Encoder    │  │   Layers    │    │
│  └─────────────┘  └─────────────┘  └─────────────┘    │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│                Multi-modal Fusion                       │
│              (Self-Attention Mechanism)                 │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│              Fused [LRN] Token Output                   │
│            (Image + Text + Action Info)                 │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│                Policy Head (LSTM)                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │   LSTM      │  │   Linear    │  │   Output    │    │
│  │  (History)  │  │   Layers    │  │ (7-DOF)     │    │
│  └─────────────┘  └─────────────┘  └─────────────┘    │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│                7-DOF Action Prediction                  │
│        [Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper]      │
└─────────────────────────────────────────────────────────┘
```

### 1.2 모듈별 역할 분담

| **모듈** | **역할** | **입력** | **출력** | **학습** |
|----------|----------|----------|----------|----------|
| **Vision Encoder** | 이미지 → 특징 벡터 | RGB 이미지 | Vision tokens | ✅ 학습 |
| **Text Encoder** | 텍스트 → 특징 벡터 | 언어 명령 | Text tokens | ✅ 학습 |
| **VLM Backbone** | Multi-modal Fusion | Text + Vision + [LRN] | Fused [LRN] | ✅ 학습 |
| **[LRN] Token** | 정보 융합 매개체 | 초기화 embedding | VLM 출력 | ✅ 학습 |
| **LSTM (Policy Head)** | 액션 예측 + 히스토리 | Fused [LRN] | 7-DOF action | ✅ 학습 |

## 2. VLM Backbone 상세 구조

### 2.1 지원하는 VLM 모델들

```python
# 지원 VLM 모델 목록
SUPPORTED_MODELS = [
    "kosmos2",      # Microsoft Kosmos-2
    "paligemma",    # Google PaLI-Gemma
    "llava",        # LLaVA
    "qwen-vl",      # Qwen-VL
    "uform",        # UForm
    "moondream",    # MoonDream
    "flamingo"      # Flamingo
]
```

### 2.2 VLM 초기화 과정

```python
# BaseRoboVLM 초기화
def __init__(self, model_name, action_space="continuous", ...):
    # 1. VLM Backbone 로드
    self.model = self._load_vlm_backbone(model_name)
    
    # 2. Vision Encoder 설정
    self.vision_tower = self._setup_vision_encoder()
    
    # 3. Action Token 생성
    if action_space == "continuous":
        self.action_token = nn.Parameter(torch.zeros(self.hidden_size))
        self.action_token.requires_grad_(True)
    
    # 4. Policy Head (LSTM) 설정
    self.act_head = self._setup_policy_head()
```

### 2.3 Multi-modal Input 생성

```python
def merge_multi_modal_input(self, text_embeds, vision_embeds, action_tokens):
    """
    Text, Vision, Action 토큰을 하나의 시퀀스로 결합
    """
    # 토큰 순서: [Text Tokens] + [Vision Tokens] + [Action Tokens]
    multimodal_embeds = torch.cat([
        text_embeds,      # [batch, text_len, hidden_size]
        vision_embeds,    # [batch, vision_len, hidden_size]  
        action_tokens     # [batch, action_len, hidden_size]
    ], dim=1)  # [batch, total_len, hidden_size]
    
    return multimodal_embeds
```

## 3. Policy Head (LSTM) 구조

### 3.1 LSTM 설정

```python
# LSTM Policy Head 설정
{
    "act_head": {
        "type": "LSTMDecoder",
        "action_space": "continuous",
        "action_dim": 7,
        "hidden_size": 1024,
        "num_layers": 2,
        "dropout": 0.1,
        "window_size": 1,              # VLM은 단일 프레임만 처리
        "with_history": true,          # LSTM이 히스토리 관리
        "history_type": "post"         # LSTM에서 처리
    }
}
```

### 3.2 LSTM Forward Pass

```python
class LSTMDecoder(nn.Module):
    def __init__(self, hidden_size, action_dim, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.action_head = nn.Linear(hidden_size, action_dim)
    
    def forward(self, fused_token, hidden_state=None):
        # LSTM Forward
        lstm_out, hidden_state = self.lstm(fused_token, hidden_state)
        
        # Action Prediction
        action = self.action_head(lstm_out)
        
        return action, hidden_state
```

### 3.3 히스토리 관리

```python
# 히스토리 관리 방식
def forward_with_history(self, vision_x, lang_x, hidden_state=None):
    # 1. 현재 프레임 처리
    fused_token = self.vlm_forward(vision_x, lang_x)
    
    # 2. LSTM에 히스토리와 함께 입력
    action, new_hidden_state = self.lstm(fused_token, hidden_state)
    
    return action, new_hidden_state
```

**히스토리 관리의 장점**:
- **시간적 일관성**: 이전 프레임의 정보를 활용
- **장기 의존성**: 복잡한 태스크에서 중요한 정보 보존
- **안정성**: 급격한 액션 변화 방지

## 4. Window Size와 Sequence 처리

### 4.1 Window Size 설정

```python
# Config 설정
{
    "window_size": 8,          # VLM 입력: 8개 프레임
    "fwd_pred_next_n": 10      # Action chunk: 10개 미래 액션 예측
}
```

**Window Size = 8**의 의미:
- 과거 7프레임 + 현재 1프레임 = 총 8프레임
- VLM은 이 8개 이미지를 모두 인코딩
- LSTM은 8개 프레임의 [LRN] 토큰을 받아 히스토리 모델링

**하지만 RoboVLMs Policy-Head 구조에서는**:
```python
{
    "act_head": {
        "window_size": 1,      # VLM 입력은 단일 프레임
        "with_history": true,  # LSTM이 히스토리 관리
        "history_type": "post" # LSTM에서 처리
    }
}
```

**핵심**:
- **VLM**: 단일 프레임만 처리 (효율적)
- **LSTM**: 여러 프레임의 [LRN] 토큰을 시간 순서대로 처리하여 히스토리 학습

## 5. 학습 가능한 파라미터 설정

### 5.1 Trainable Parameters Setup

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

### 5.2 Full Fine-tuning 설정

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

## 6. 메모리 효율성 고려사항

### 6.1 Gradient Checkpointing

```python
# 메모리 효율성을 위한 Gradient Checkpointing
if self.gradient_checkpointing:
    self.model.gradient_checkpointing_enable()
```

### 6.2 Mixed Precision Training

```python
# Mixed Precision Training 설정
{
    "mixed_precision": true,
    "fp16": true,
    "bf16": false
}
```

### 6.3 Batch Size 조정

```python
# 메모리에 따른 Batch Size 조정
{
    "batch_size": 8,        # 기본값
    "gradient_accumulation_steps": 4,  # 메모리 부족시 사용
    "effective_batch_size": 32         # 8 * 4 = 32
}
```

## 7. 핵심 아키텍처 특징

### 7.1 End-to-End 학습
- VLM과 LSTM이 동시에 학습
- Gradient가 전체 파이프라인을 통해 전파
- Multi-modal 정보가 자연스럽게 융합

### 7.2 Modular Design
- VLM Backbone은 교체 가능
- Policy Head는 독립적으로 설계
- 설정 파일로 쉽게 변경 가능

### 7.3 Scalability
- 다양한 VLM 모델 지원
- 다양한 로봇 플랫폼에 적용 가능
- 실시간 추론 최적화

## 8. 참고 자료

- `RoboVLMs/robovlms/model/backbone/base_backbone.py`: 기본 VLM + LSTM 통합
- `RoboVLMs/robovlms/model/backbone/kosmos2_backbone.py`: Kosmos-2 구현
- `RoboVLMs/robovlms/model/backbone/paligemma_backbone.py`: PaLI-Gemma 구현
- `RoboVLMs/configs/calvin_finetune/`: CALVIN 학습 설정
