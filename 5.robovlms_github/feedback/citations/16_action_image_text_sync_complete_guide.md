# 16. Action-Image-Text Synchronization 완전 가이드

##  개요

이 문서는 RoboVLMs에서 **Action, Image, Text가 어떻게 동기화(Sync)되는지**를 처음부터 끝까지 순서대로 설명합니다.

---

##  핵심 질문

1. **VLM Finetuning (F-FT)과 LoRA는 무엇인가?**
2. **action과 rel_action은 어떻게 다르고, 어떻게 동기화되는가?**
3. **7-DOF 로봇팔 움직임이 어떻게 표현되고 학습되는가?**
4. **Image, Text, Action이 어떻게 동시에 학습되는가?**
5. **Embedded Token이 무엇이고 어떻게 동기화되는가?**
6. **CALVIN 데이터셋은 어떻게 구성되어 있는가?**
7. **실제 학습 과정에서 VLM과 Action Head는 동시에 학습되는가?**

---

##  Part 1: 기본 개념 이해

### 1.1 action vs rel_action: 핵심 차이

#### **1.1.1 정의**

```python
# CALVIN 데이터셋 관측 공간
obs_config = DictConfig({
    "rgb_obs": ["rgb_static", "rgb_gripper"],    # 이미지
    "state_obs": ["robot_obs"],                  # 로봇 상태 (현재 위치/자세)
    "actions": ["rel_actions"],                  #  상대적 액션 (relative actions)
    "language": ["language"],                    # 텍스트 명령
})
```
**출처**: `RoboVLMs/robovlms/data/calvin_dataset.py:63-71`

#### **1.1.2 두 가지 액션 표현 방식**

| **구분** | **action (절대 액션)** | **rel_action (상대 액션)** |
|---------|---------------------|------------------------|
| **좌표계** | World Frame (고정된 global 좌표) | TCP Frame (End-Effector 기준) |
| **의미** | "책상 위 (x=0.5, y=0.3, z=0.2) 위치로 이동" | "현재 위치에서 오른쪽으로 5cm 이동" |
| **데이터** | 절대 좌표 값 | 현재 위치 대비 상대 변화량 |
| **사용** | 시뮬레이션, 고정 환경 | **RoboVLMs, CALVIN (표준)** |
| **장점** | 명확한 목표 지점 | 일반화 가능, 로봇 위치 무관 |

**출처**: `RoboVLMs/robovlms/data/calvin_dataset.py:192`

---

#### **1.1.3 상대 액션 (rel_action)의 구성: 7-DOF**

```python
# 7-DOF 상대 액션 구조
rel_action = [
    # Translation (3-DOF): TCP Frame 기준 상대 이동
    Δx,  # TCP 기준 앞/뒤 (forward/backward)
    Δy,  # TCP 기준 좌/우 (left/right)
    Δz,  # TCP 기준 위/아래 (up/down)
    
    # Rotation (3-DOF): TCP Frame 기준 상대 회전 (Euler angles)
    Δroll,   # TCP 기준 X축 회전
    Δpitch,  # TCP 기준 Y축 회전
    Δyaw,    # TCP 기준 Z축 회전
    
    # Gripper (1-DOF): 그리퍼 열기/닫기
    gripper  # ∈ {-1 (open), 1 (close)}
]
```

**핵심**: 
- **절대 위치 7개가 아니라, 현재 상태 대비 변화량 7개**
- 로봇이 어디에 있든 "오른쪽으로 5cm"는 동일한 의미
- 일반화 성능이 높음

---

#### **1.1.4 World Frame ↔ TCP Frame 변환**

**World to TCP Frame 변환** (데이터 전처리):
```python
def world_to_tcp_frame(action, robot_obs):
    """
    World frame의 action을 TCP frame의 rel_action으로 변환
    """
    # 1. 현재 로봇 자세 (robot_obs)에서 World → TCP 변환 행렬 생성
    world_T_tcp = euler_angles_to_matrix(robot_obs[..., 3:6], convention="XYZ")
    tcp_T_world = torch.inverse(world_T_tcp)
    
    # 2. Translation 변환 (World → TCP)
    pos_w_rel = action[..., :3]  # World frame에서의 이동량
    pos_tcp_rel = tcp_T_world @ pos_w_rel  # TCP frame으로 변환
    
    # 3. Rotation 변환 (World → TCP)
    # Downscaling: 0.01배 (pseudo infinitesimal rotation)
    orn_w_rel = action[..., 3:6] * 0.01
    world_T_tcp_new = euler_angles_to_matrix(
        robot_obs[..., 3:6] + orn_w_rel, convention="XYZ"
    )
    tcp_new_T_tcp_old = torch.inverse(world_T_tcp_new) @ world_T_tcp
    orn_tcp_rel = matrix_to_euler_angles(tcp_new_T_tcp_old, convention="XYZ")
    
    # Upscaling: 100배 (원래 스케일로 복원)
    orn_tcp_rel *= 100
    
    # 4. 최종 rel_action 생성
    action_tcp = torch.cat([
        pos_tcp_rel,      # TCP frame 기준 이동
        orn_tcp_rel,      # TCP frame 기준 회전
        action[..., -1:]  # Gripper (변환 불필요)
    ], dim=-1)
    
    return action_tcp
```
**출처**: `RoboVLMs/robovlms/data/data_utils.py:770-820`

**핵심**:
- **Rotation downscaling/upscaling**: 작은 회전 변화를 정확히 표현하기 위한 수치적 안정성 기법
- 0.01배 → 변환 → 100배로 원복
- **Translation은 직접 변환**, **Rotation은 행렬 기반 변환**

---

#### **1.1.5 정규화: [-1, 1] 범위로 Clipping**

```python
# Action Normalization (논문 기준)
# 1단계: 1st, 99th percentile 기반 clipping
ai′ = min(ai_99th, max(ai_1st, ai))

# 2단계: [-1, 1] 정규화
ãi = 2 × (ai′ − ai_1st) / (ai_99th − ai_1st) − 1

# CALVIN 데이터셋 설정
{
    "norm_action": true,
    "norm_min": -0.65,  # 1st percentile
    "norm_max": 0.65    # 99th percentile
}
```
**출처**: 
- RoboVLMs 논문 Equation (5), (6)
- `RoboVLMs/configs/calvin_finetune/finetune_kosmos_cont-lstm-post_full-ft_text_vision_wd-0_ws-8_act-10.json:126-128`

**왜 [-1, 1]인가?**
1. **학습 안정성**: 신경망은 0 근처 값에서 gradient가 안정적
2. **Tanh/Sigmoid 출력**: LSTM의 출력 활성화 함수와 일치
3. **클리핑 효과**: 이상치(outlier) 제거

---

### 1.2 VLM Fine-tuning (F-FT)과 LoRA

#### **1.2.1 Fine-tuning (F-FT) 개념**

```python
# Full Fine-tuning 설정
{
    "train_setup": {
        "freeze_backbone": false,      # VLM 전체 학습
        "train_vision": true,          # Vision Encoder 학습
        "train_text_embedding": true,  # Text Embedding 학습
        "lora_enable": false           # LoRA 비활성화 (Full-FT)
    }
}
```
**출처**: `RoboVLMs/configs/calvin_finetune/finetune_kosmos_cont-lstm-post_full-ft_text_vision_wd-0_ws-8_act-10.json:44-64`

**Full Fine-tuning의 의미**:
- VLM의 **모든 파라미터**를 로봇 태스크에 맞춰 재학습
- Vision Encoder + Language Model 전체 업데이트
- 메모리와 계산량이 많지만, **최고 성능** 달성

---

#### **1.2.2 LoRA (Low-Rank Adaptation)**

```python
# LoRA 설정 (메모리 효율적 fine-tuning)
{
    "train_setup": {
        "lora_enable": true,     # LoRA 활성화
        "lora_r": 64,            # Rank (저차원 행렬 크기)
        "lora_alpha": 16,        # Scaling factor
        "lora_dropout": 0.05,    # Dropout rate
        "lora_bias": "none"      # Bias 학습 안 함
    }
}
```
**출처**: `RoboVLMs/README.md:245-248`

**LoRA의 원리**:
```
Original Weight: W ∈ R^(d×k)
LoRA: ΔW = B×A (B ∈ R^(d×r), A ∈ R^(r×k), r << d)
Updated Weight: W' = W + α·(B×A)
```
- 원본 가중치 `W`는 고정 (frozen)
- 저차원 행렬 `B`, `A`만 학습 (파라미터 수: `r×(d+k)`)
- `r=64`일 때, 전체 파라미터의 **0.1% 미만**만 학습

**장점**:
- 메모리 효율적 (GPU VRAM 절약)
- 빠른 학습 속도
- 성능은 Full-FT의 90~95% 수준

**RoboVLMs 선택**: **Full-FT** (최고 성능 우선)

---

### 1.3 Embedded Token: Multi-modal Fusion의 핵심

#### **1.3.1 Token의 정의**

**Token**: VLM이 처리하는 **모든 입력의 기본 단위**

```python
# VLM 입력 토큰 시퀀스
input_sequence = [
    # 1. Text Tokens (언어 명령)
    [BOS], "pick", "up", "the", "red", "block", [EOS],
    
    # 2. Vision Tokens (이미지 특징)
    [IMG_1], [IMG_2], ..., [IMG_N],  # N개 patch tokens
    
    # 3. Action Token (학습 가능한 토큰)
    [LRN]  # Learnable action token
]
```

**핵심**: 텍스트, 이미지, 액션이 **모두 같은 토큰 형태**로 변환되어 VLM에 입력됨

---

#### **1.3.2 Action Token ([LRN]): Embedding으로 Multi-modal 정보 융합**

```python
# BaseRoboVLM 초기화
def __init__(self, ...):
    if self.action_space == "continuous":
        # 학습 가능한 Action Token 생성
        self.action_token = nn.Parameter(torch.zeros(self.hidden_size))
        self.action_token.requires_grad_(True)  #  학습 가능
```
**출처**: `RoboVLMs/robovlms/model/backbone/base_backbone.py:124-126`

**[LRN] 토큰의 역할**:
1. **초기값**: 0으로 초기화된 embedding vector (shape: `[hidden_size]`, 예: `[1024]`)
2. **VLM 통과**: Text tokens + Vision tokens + [LRN] 모두 VLM을 통과
3. **Multi-modal Fusion**: VLM 내부의 self-attention/cross-attention을 통해 **이미지와 텍스트 정보가 [LRN]에 융합**
4. **출력**: VLM이 [LRN] 토큰에 "이 이미지와 이 명령에 필요한 액션 정보"를 인코딩
5. **Policy Head 입력**: 융합된 [LRN]을 LSTM에 입력하여 7-DOF 액션 예측

---

#### **1.3.3 Token Synchronization 과정**

```python
# forward_continuous() - 토큰 동기화 핵심 코드
def forward_continuous(self, vision_x, lang_x, ...):
    # 1. Text Tokens 생성
    text_embeds = self.word_embedding(lang_x)  # shape: [batch, seq_len, hidden_size]
    
    # 2. Vision Tokens 생성
    vision_embeds = self.encode_images(vision_x)  # shape: [batch, num_patches, hidden_size]
    
    # 3. Action Token 추가
    action_tokens = repeat(
        self.action_token,  # [hidden_size]
        "d -> b n d",
        b=batch_size,
        n=self.latent_num  # 보통 1
    )  # shape: [batch, 1, hidden_size]
    
    # 4. Multi-modal Fusion (VLM에 입력)
    multimodal_embeds = self.merge_multi_modal_input(
        text_embeds,      # Text tokens
        vision_embeds,    # Vision tokens
        action_tokens     # Action token
    )  # shape: [batch, total_len, hidden_size]
    
    # 5. VLM Backbone 통과
    vlm_output = self.model(inputs_embeds=multimodal_embeds, ...)
    
    # 6. [LRN] 토큰 추출 (마지막 토큰)
    action_token_output = vlm_output[:, -1, :]  # shape: [batch, hidden_size]
    
    # 7. Policy Head (LSTM) 입력
    predicted_action = self.act_head(action_token_output)  # shape: [batch, 7]
    
    return predicted_action
```
**출처**: `RoboVLMs/robovlms/model/backbone/base_backbone.py` (통합 설명)

**동기화 핵심**:
- **같은 공간**: Text, Vision, Action 모두 `hidden_size` 차원의 embedding
- **Attention 메커니즘**: VLM의 self-attention이 자동으로 관련성 학습
- **학습 과정**: Action Token의 embedding이 점진적으로 최적화됨

---

##  Part 2: CALVIN 데이터셋 구조

### 2.1 CALVIN 데이터셋 개요

#### **2.1.1 데이터 구성**

```python
# CALVIN 샘플 구조
{
    "rgb_static": np.array([224, 224, 3]),     # 정적 카메라 이미지
    "rgb_gripper": np.array([224, 224, 3]),    # 그리퍼 카메라 이미지
    "robot_obs": np.array([15]),               # 로봇 상태 (joint angles, TCP pose 등)
    "rel_actions": np.array([7]),              # 상대 액션 (7-DOF)
    "language": str,                           # 언어 명령 (예: "pick up the red block")
}
```
**출처**: `RoboVLMs/robovlms/data/calvin_dataset.py:63-71`

---

#### **2.1.2 robot_obs (로봇 상태) 상세 구조**

```python
# robot_obs: 15차원 벡터
robot_obs = [
    # TCP Pose (7차원)
    tcp_x, tcp_y, tcp_z,           # TCP 위치 (World frame)
    tcp_roll, tcp_pitch, tcp_yaw,  # TCP 자세 (Euler angles)
    gripper_state,                 # Gripper 상태 (0~1)
    
    # Joint Angles (7차원) - Franka Emika Panda 로봇
    joint_1, joint_2, ..., joint_7,
    
    # Gripper Width (1차원)
    gripper_width
]
```

**핵심**: 
- `robot_obs[3:6]` (TCP 자세)를 사용하여 World ↔ TCP frame 변환
- 현재 상태를 알아야 rel_action을 적용 가능

---

#### **2.1.3 데이터 전처리 파이프라인**

```python
# CALVIN Dataset __getitem__() 핵심 로직
def __getitem__(self, idx):
    # 1. 원본 데이터 로드
    rgb_static = load_image(episode["rgb_static"])
    rgb_gripper = load_image(episode["rgb_gripper"])
    robot_obs = episode["robot_obs"]
    actions = episode["rel_actions"]  # 이미 rel_action
    language = episode["language"]
    
    # 2. 이미지 전처리
    rgb_static = self.transforms(rgb_static)  # Resize, Normalize
    rgb_gripper = self.transforms(rgb_gripper)
    
    # 3. Action 정규화
    actions = self.normalize_action(actions)  # [-1, 1] 범위로
    
    # 4. 데이터 반환
    return {
        "vision_x": torch.stack([rgb_static, rgb_gripper]),  # [2, 3, 224, 224]
        "lang_x": self.tokenizer.encode(language),           # [seq_len]
        "action_labels": actions,                            # [7]
        "robot_obs": robot_obs                               # [15]
    }
```
**출처**: `RoboVLMs/robovlms/data/calvin_dataset.py` (통합 설명)

---

### 2.2 Window Size와 Sequence 처리

#### **2.2.1 Window Size의 의미**

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
**출처**: `RoboVLMs/configs/calvin_finetune/finetune_kosmos_cont-lstm-post_full-ft_text_vision_wd-0_ws-8_act-10.json:74-84`

**핵심**: 
- **VLM**: 단일 프레임만 처리 (효율적)
- **LSTM**: 여러 프레임의 [LRN] 토큰을 시간 순서대로 처리하여 히스토리 학습

---

##  Part 3: 학습 과정 - VLM과 Action Head 동시 학습

### 3.1 학습 파이프라인 전체 흐름

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

---

### 3.2 Forward Pass 상세

```python
# BaseRoboVLM.forward_continuous() 핵심
def forward_continuous(self, vision_x, lang_x, action_labels, ...):
    # 1. Text Embedding
    text_embeds = self.word_embedding(lang_x)  # [batch, text_len, 1024]
    
    # 2. Vision Encoding (VLM Vision Encoder)
    vision_embeds = self.encode_images(vision_x)  # [batch, num_patches, 1024]
    
    # 3. Action Token 추가
    action_tokens = repeat(self.action_token, "d -> b n d", b=batch_size, n=1)
    
    # 4. Multi-modal Input 생성
    multimodal_embeds = torch.cat([
        text_embeds,     # Text tokens
        vision_embeds,   # Vision tokens
        action_tokens    # [LRN] token
    ], dim=1)  # [batch, total_len, 1024]
    
    # 5. VLM Backbone 통과 (Kosmos, PaliGemma, etc.)
    vlm_output = self.model(inputs_embeds=multimodal_embeds, ...)
    # [batch, total_len, 1024]
    
    # 6. [LRN] Token 추출 (마지막 토큰)
    action_token_output = vlm_output[:, -1, :]  # [batch, 1024]
    
    # 7. Policy Head (LSTM) Forward
    predicted_action = self.act_head(action_token_output)  # [batch, 7]
    
    # 8. Loss 계산
    loss_pose = F.mse_loss(predicted_action[:, :6], action_labels[:, :6])
    loss_gripper = F.binary_cross_entropy_with_logits(
        predicted_action[:, 6], action_labels[:, 6]
    )
    loss = loss_pose + 0.01 * loss_gripper
    
    return loss
```
**출처**: `RoboVLMs/robovlms/model/backbone/base_backbone.py` (통합)

---

### 3.3 Loss Function과 Gradient Flow

#### **3.3.1 Loss 계산**

```python
# Continuous Action Loss (논문 Equation 7)
lVLA = Σ [MSE(ât,pose, ãt,pose) + λ * BCE(ât,gripper, ãt,gripper)]

# 구현
loss_pose = F.mse_loss(predicted_action[:, :6], ground_truth[:, :6])
loss_gripper = F.binary_cross_entropy_with_logits(
    predicted_action[:, 6], ground_truth[:, 6]
)
total_loss = loss_pose + lambda_gripper * loss_gripper  # lambda_gripper = 0.01
```
**출처**: RoboVLMs 논문 Equation (7)

**왜 MSE + BCE?**
- **MSE (처음 6차원)**: Translation + Rotation은 연속 값 (회귀 문제)
- **BCE (마지막 1차원)**: Gripper는 binary (열기 -1 / 닫기 1)
- **λ = 0.01**: Gripper loss의 가중치를 낮춰 pose 학습 우선

---

#### **3.3.2 Gradient Flow: VLM과 LSTM 동시 학습**

```
Loss
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

---

### 3.4 VLM과 Action Head의 역할 분담

| **모듈** | **역할** | **입력** | **출력** | **학습** |
|---------|---------|---------|---------|---------|
| **Vision Encoder** | 이미지 → 특징 벡터 | RGB 이미지 | Vision tokens |  학습 |
| **Text Encoder** | 텍스트 → 특징 벡터 | 언어 명령 | Text tokens |  학습 |
| **VLM Backbone** | Multi-modal Fusion | Text + Vision + [LRN] | Fused [LRN] |  학습 |
| **[LRN] Token** | 정보 융합 매개체 | 초기화 embedding | VLM 출력 |  학습 |
| **LSTM (Policy Head)** | 액션 예측 + 히스토리 | Fused [LRN] | 7-DOF action |  학습 |

**결론**: **VLM과 Action Head는 동시에 학습됩니다** (End-to-End)

---

##  Part 4: 실제 학습 설정 (Kosmos + LSTM)

### 4.1 Full Fine-tuning Config

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
    },
    "act_head": {
        "type": "LSTMDecoder",
        "action_space": "continuous",
        "action_dim": 7,
        "down_sample": "none",
        "window_size": 1,              // VLM은 단일 프레임만 처리
        "with_history": true,          // LSTM이 히스토리 관리
        "history_type": "post"
    }
}
```
**출처**: `RoboVLMs/configs/calvin_finetune/finetune_kosmos_cont-lstm-post_full-ft_text_vision_wd-0_ws-8_act-10.json`

---

### 4.2 Trainable Parameters Setup

```python
def _trainable_params_setup(self):
    model = self.model  # VLM Backbone
    
    # 1. 백본 모델 설정
    if self.train_setup_configs["freeze_backbone"]:
        model.requires_grad_(False)  # 동결
    else:
        model.requires_grad_(True)   #  전체 학습
    
    # 2. Vision Encoder 설정
    if self.train_setup_configs.get("train_vision", False):
        self.vision_tower.requires_grad_(True)  #  학습
    else:
        self.vision_tower.requires_grad_(False)
    
    # 3. LoRA 설정 (Full-FT는 skip)
    if self.train_setup_configs["lora_enable"]:
        # LoRA 적용 (RoboVLMs는 사용 안 함)
        pass
    
    # 4. Action Token 학습 (항상 True)
    self.action_token.requires_grad_(True)  #  학습
    
    # 5. Policy Head 학습 (항상 True)
    self.act_head.requires_grad_(True)  #  학습
```
**출처**: `RoboVLMs/robovlms/model/backbone/base_backbone.py` (통합)

**결론**:
- **VLM Vision Encoder**:  학습
- **VLM Text Encoder**:  학습
- **VLM Backbone (Attention)**:  학습
- **[LRN] Token**:  학습
- **LSTM Policy Head**:  학습

**모든 것이 동시에 학습됩니다!**

---

##  최종 정리: 전체 플로우

### Step 1: 데이터 준비
```
CALVIN Dataset
  ↓
Image (rgb_static, rgb_gripper) + Text (language) + robot_obs + rel_actions
  ↓
전처리: Image Resize/Normalize, Action Normalize [-1, 1]
```

### Step 2: Forward Pass
```
Image → Vision Encoder → Vision Tokens
Text → Tokenizer → Text Tokens
[LRN] Token 생성 (학습 가능)
  ↓
Multi-modal Fusion (VLM Backbone)
  Text Tokens + Vision Tokens + [LRN] → Attention → Fused [LRN]
  ↓
LSTM Policy Head
  Fused [LRN] → LSTM → 7-DOF Action Prediction
```

### Step 3: Loss & Backpropagation
```
Predicted Action vs Ground Truth rel_action
  ↓
Loss = MSE(pose) + λ * BCE(gripper)
  ↓
Backpropagation: LSTM → VLM → Vision/Text Encoder
  ↓
모든 파라미터 동시 업데이트 (End-to-End)
```

### Step 4: 결과
```
Vision Encoder: 로봇 태스크에 유용한 이미지 특징 학습
Text Encoder: 명령과 액션의 관계 학습
[LRN] Token: Multi-modal 정보 융합 학습
LSTM: 7-DOF 액션 예측 + 히스토리 모델링
```

---

##  핵심 Q&A 요약

### Q1. VLM Finetuning과 LoRA는 무엇인가?
**A**: 
- **Full-FT**: VLM 전체 파라미터 재학습 (RoboVLMs 사용)
- **LoRA**: 저차원 행렬만 학습 (메모리 효율적, 성능 약간 낮음)

### Q2. action과 rel_action의 차이는?
**A**:
- **action**: World frame 절대 좌표
- **rel_action**: TCP frame 상대 변화량 (RoboVLMs 사용)
- **변환**: `world_to_tcp_frame()` 함수로 변환

### Q3. 7-DOF는 어떻게 표현되는가?
**A**: Translation(3) + Rotation(3) + Gripper(1) = 7차원 벡터

### Q4. Image, Text, Action은 어떻게 동기화되는가?
**A**: 
- 모두 **Token**으로 변환 → VLM Attention으로 융합
- **[LRN] Token**이 Multi-modal 정보 통합

### Q5. Embedded Token이란?
**A**: 
- **[LRN]**: 학습 가능한 Action Token
- VLM을 통과하며 Image + Text 정보를 융합

### Q6. CALVIN 데이터셋 구조는?
**A**: 
- Image(2개) + Text + robot_obs(15차원) + rel_actions(7차원)
- 24K demonstrations, 34 basic skills

### Q7. VLM과 Action Head는 동시에 학습되는가?
**A**: **예!** End-to-End로 모든 파라미터 동시 학습

---

##  참고 자료

- RoboVLMs 논문: Section B (VLA Models), Section C (VLA Structures)
- `RoboVLMs/robovlms/data/calvin_dataset.py`: CALVIN 데이터 로더
- `RoboVLMs/robovlms/data/data_utils.py`: World ↔ TCP frame 변환
- `RoboVLMs/robovlms/model/backbone/base_backbone.py`: VLM + LSTM 통합
- `RoboVLMs/configs/calvin_finetune/`: CALVIN 학습 설정

