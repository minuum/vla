# Action-Image-Text Synchronization 핵심 개념

## 1. Action vs Rel_Action: 핵심 차이점

### 1.1 정의

**Action (절대 액션)**:
- World Frame 기준 절대 좌표
- "책상 위 (x=0.5, y=0.3, z=0.2) 위치로 이동"
- 고정된 환경에서만 사용 가능

**Rel_Action (상대 액션)**:
- TCP Frame 기준 상대 변화량
- "현재 위치에서 오른쪽으로 5cm 이동"
- 로봇 위치에 무관하게 일반화 가능

### 1.2 7-DOF 상대 액션 구조

```python
rel_action = [
    # Translation (3-DOF): TCP Frame 기준 상대 이동
    Δx,      # TCP 기준 앞/뒤 (forward/backward)
    Δy,      # TCP 기준 좌/우 (left/right)  
    Δz,      # TCP 기준 위/아래 (up/down)
    
    # Rotation (3-DOF): TCP Frame 기준 상대 회전 (Euler angles)
    Δroll,   # TCP 기준 X축 회전
    Δpitch,  # TCP 기준 Y축 회전
    Δyaw,    # TCP 기준 Z축 회전
    
    # Gripper (1-DOF): 그리퍼 열기/닫기
    gripper  # ∈ {-1 (open), 1 (close)}
]
```

### 1.3 World Frame ↔ TCP Frame 변환

**핵심 변환 함수**: `world_to_tcp_frame()`

```python
def world_to_tcp_frame(action, robot_obs):
    """
    World frame의 action을 TCP frame의 rel_action으로 변환
    """
    # 1. 현재 로봇 자세에서 World → TCP 변환 행렬 생성
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

**핵심 포인트**:
- **Rotation downscaling/upscaling**: 작은 회전 변화를 정확히 표현하기 위한 수치적 안정성 기법
- 0.01배 → 변환 → 100배로 원복
- **Translation은 직접 변환**, **Rotation은 행렬 기반 변환**

### 1.4 Action 정규화: [-1, 1] 범위로 Clipping

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

**왜 [-1, 1]인가?**
1. **학습 안정성**: 신경망은 0 근처 값에서 gradient가 안정적
2. **Tanh/Sigmoid 출력**: LSTM의 출력 활성화 함수와 일치
3. **클리핑 효과**: 이상치(outlier) 제거

## 2. VLM Fine-tuning (F-FT)과 LoRA

### 2.1 Full Fine-tuning (F-FT)

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

**Full Fine-tuning의 의미**:
- VLM의 **모든 파라미터**를 로봇 태스크에 맞춰 재학습
- Vision Encoder + Language Model 전체 업데이트
- 메모리와 계산량이 많지만, **최고 성능** 달성

### 2.2 LoRA (Low-Rank Adaptation)

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

## 3. Embedded Token: Multi-modal Fusion의 핵심

### 3.1 Token의 정의

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

### 3.2 Action Token ([LRN]): Embedding으로 Multi-modal 정보 융합

```python
# BaseRoboVLM 초기화
def __init__(self, ...):
    if self.action_space == "continuous":
        # 학습 가능한 Action Token 생성
        self.action_token = nn.Parameter(torch.zeros(self.hidden_size))
        self.action_token.requires_grad_(True)  # ⭐ 학습 가능
```

**[LRN] 토큰의 역할**:
1. **초기값**: 0으로 초기화된 embedding vector (shape: `[hidden_size]`, 예: `[1024]`)
2. **VLM 통과**: Text tokens + Vision tokens + [LRN] 모두 VLM을 통과
3. **Multi-modal Fusion**: VLM 내부의 self-attention/cross-attention을 통해 **이미지와 텍스트 정보가 [LRN]에 융합**
4. **출력**: VLM이 [LRN] 토큰에 "이 이미지와 이 명령에 필요한 액션 정보"를 인코딩
5. **Policy Head 입력**: 융합된 [LRN]을 LSTM에 입력하여 7-DOF 액션 예측

### 3.3 Token Synchronization 과정

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

**동기화 핵심**:
- **같은 공간**: Text, Vision, Action 모두 `hidden_size` 차원의 embedding
- **Attention 메커니즘**: VLM의 self-attention이 자동으로 관련성 학습
- **학습 과정**: Action Token의 embedding이 점진적으로 최적화됨

## 4. 핵심 Q&A 요약

### Q1. action과 rel_action의 차이는?
**A**: 
- **action**: World frame 절대 좌표
- **rel_action**: TCP frame 상대 변화량 (RoboVLMs 사용)
- **변환**: `world_to_tcp_frame()` 함수로 변환

### Q2. 7-DOF는 어떻게 표현되는가?
**A**: Translation(3) + Rotation(3) + Gripper(1) = 7차원 벡터

### Q3. Image, Text, Action은 어떻게 동기화되는가?
**A**: 
- 모두 **Token**으로 변환 → VLM Attention으로 융합
- **[LRN] Token**이 Multi-modal 정보 통합

### Q4. Embedded Token이란?
**A**: 
- **[LRN]**: 학습 가능한 Action Token
- VLM을 통과하며 Image + Text 정보를 융합

### Q5. VLM Finetuning과 LoRA는 무엇인가?
**A**:
- **Full-FT**: VLM 전체 파라미터 재학습 (RoboVLMs 사용)
- **LoRA**: 저차원 행렬만 학습 (메모리 효율적, 성능 약간 낮음)
