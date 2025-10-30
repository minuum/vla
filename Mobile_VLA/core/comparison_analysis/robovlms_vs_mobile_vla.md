# RoboVLMs vs Mobile VLA 비교 분석

## 1. 전체 아키텍처 비교

### 1.1 기존 RoboVLMs (GitHub)

**아키텍처**:
```
Input: [Image, Text] → VLM (Kosmos/PaLI) → [LRN] Token → LSTM → 7-DOF Action
```

**특징**:
- **Full Fine-tuning**: VLM 전체 파라미터 학습
- **CALVIN 데이터셋**: 24K demonstrations
- **7-DOF 액션**: Translation(3) + Rotation(3) + Gripper(1)
- **Window Size**: 8 (히스토리 길이)
- **Action Chunk**: 10 (미래 액션 예측)

### 1.2 우리 Mobile VLA

**아키텍처**:
```
Input: [Image, Text] → VLM (Kosmos) → [LRN] Token → LSTM → 2D Action
```

**특징**:
- **LoRA Fine-tuning**: 메모리 효율적 학습
- **Custom Dataset**: 1000개 demonstrations (목표)
- **2D 액션**: X(1) + Y(1) + Gripper(1) = 3차원
- **Window Size**: 8 (동일)
- **Action Chunk**: 10 (동일)

## 2. 학습 방법 비교

### 2.1 기존 RoboVLMs 학습

**Full Fine-tuning 설정**:
```json
{
    "train_setup": {
        "freeze_backbone": false,      // VLM 전체 학습
        "train_vision": true,          // Vision Encoder 학습
        "train_text_embedding": true,  // Text Embedding 학습
        "lora_enable": false,          // LoRA 비활성화
        "learning_rate": 2e-5,
        "weight_decay": 0
    }
}
```

**메모리 요구사항**:
- **GPU Memory**: 24GB+ (RTX 4090, A100)
- **Batch Size**: 4-8
- **Precision**: bf16
- **Gradient Checkpointing**: True

### 2.2 우리 Mobile VLA 학습

**LoRA Fine-tuning 설정**:
```json
{
    "train_setup": {
        "freeze_backbone": true,       // VLM 백본 동결
        "train_vision": false,         // Vision Encoder 동결
        "train_text_embedding": false, // Text Embedding 동결
        "lora_enable": true,           // LoRA 활성화
        "lora_r": 64,                  // LoRA rank
        "lora_alpha": 16,              // LoRA scaling
        "learning_rate": 1e-4,         // 높은 학습률
        "weight_decay": 0.01
    }
}
```

**메모리 요구사항**:
- **GPU Memory**: 8-16GB (Jetson AGX Orin)
- **Batch Size**: 2-4
- **Precision**: fp16
- **Gradient Checkpointing**: True

## 3. 데이터셋 비교

### 3.1 기존 RoboVLMs 데이터셋

**CALVIN 데이터셋**:
- **규모**: 24,000 demonstrations
- **로봇**: Franka Emika Panda 7-DOF
- **환경**: 시뮬레이션 (MuJoCo)
- **태스크**: 34개 기본 스킬
- **액션**: 7-DOF (World frame → TCP frame 변환)

**데이터 구조**:
```python
{
    "rgb_static": [224, 224, 3],      # 정적 카메라
    "rgb_gripper": [224, 224, 3],     # 그리퍼 카메라
    "robot_obs": [15],                # 로봇 상태
    "rel_actions": [7],               # 7-DOF 액션
    "language": str                   # 언어 명령
}
```

### 3.2 우리 Mobile VLA 데이터셋

**Custom 데이터셋**:
- **규모**: 1,000 demonstrations (목표)
- **로봇**: Mobile Robot (2D 이동)
- **환경**: 실제 환경 (Real-world)
- **태스크**: Mobile navigation + manipulation
- **액션**: 2D (X, Y, Gripper)

**데이터 구조**:
```python
{
    "rgb_static": [224, 224, 3],      # 정적 카메라
    "rgb_gripper": [224, 224, 3],     # 그리퍼 카메라 (선택적)
    "robot_obs": [3],                 # 2D 위치 + 그리퍼 상태
    "rel_actions": [3],               # 2D 액션 (X, Y, Gripper)
    "language": str                   # 언어 명령
}
```

## 4. 액션 공간 비교

### 4.1 기존 RoboVLMs 액션 공간

**7-DOF 액션**:
```python
action = [
    # Translation (3-DOF)
    delta_x,    # TCP 기준 X 이동
    delta_y,    # TCP 기준 Y 이동
    delta_z,    # TCP 기준 Z 이동
    
    # Rotation (3-DOF)
    delta_roll,   # TCP 기준 Roll 회전
    delta_pitch,  # TCP 기준 Pitch 회전
    delta_yaw,    # TCP 기준 Yaw 회전
    
    # Gripper (1-DOF)
    gripper      # 그리퍼 열기/닫기
]
```

**정규화**:
```python
# CALVIN 표준 정규화
norm_min, norm_max = -0.65, 0.65
normalized_action = np.clip(action, norm_min, norm_max)
```

### 4.2 우리 Mobile VLA 액션 공간

**2D 액션**:
```python
action = [
    # 2D Movement (2-DOF)
    delta_x,    # X 방향 이동
    delta_y,    # Y 방향 이동
    
    # Gripper (1-DOF)
    gripper     # 그리퍼 열기/닫기
]
```

**정규화**:
```python
# Mobile VLA 정규화
norm_min, norm_max = -1.0, 1.0
normalized_action = np.clip(action, norm_min, norm_max)
```

## 5. 학습 파이프라인 비교

### 5.1 기존 RoboVLMs 학습 파이프라인

**Full Fine-tuning**:
```python
# 1. VLM 전체 학습
for param in vlm.parameters():
    param.requires_grad = True

# 2. LSTM 학습
for param in lstm.parameters():
    param.requires_grad = True

# 3. End-to-End 학습
loss = vlm_loss + lstm_loss
loss.backward()
optimizer.step()
```

**메모리 사용량**:
- **VLM 파라미터**: ~1B (Kosmos-2)
- **LSTM 파라미터**: ~10M
- **총 메모리**: 24GB+

### 5.2 우리 Mobile VLA 학습 파이프라인

**LoRA Fine-tuning**:
```python
# 1. VLM 백본 동결
for param in vlm.parameters():
    param.requires_grad = False

# 2. LoRA 파라미터만 학습
for param in lora_parameters:
    param.requires_grad = True

# 3. LSTM 학습
for param in lstm.parameters():
    param.requires_grad = True

# 4. LoRA + LSTM 학습
loss = lora_loss + lstm_loss
loss.backward()
optimizer.step()
```

**메모리 사용량**:
- **VLM 파라미터**: 0 (동결)
- **LoRA 파라미터**: ~1M
- **LSTM 파라미터**: ~10M
- **총 메모리**: 8-16GB

## 6. 추론 성능 비교

### 6.1 기존 RoboVLMs 추론

**추론 설정**:
```python
# Full VLM 추론
model.eval()
with torch.no_grad():
    action = model(image, text)
```

**성능**:
- **정확도**: 높음 (Full Fine-tuning)
- **속도**: 중간 (큰 모델)
- **메모리**: 8-12GB

### 6.2 우리 Mobile VLA 추론

**추론 설정**:
```python
# LoRA + LSTM 추론
model.eval()
with torch.no_grad():
    action = model(image, text)
```

**성능**:
- **정확도**: 중간 (LoRA Fine-tuning)
- **속도**: 빠름 (작은 모델)
- **메모리**: 4-8GB

## 7. 컨테이너 및 배포 비교

### 7.1 기존 RoboVLMs 배포

**Docker 설정**:
```dockerfile
FROM nvidia/cuda:11.8-devel-ubuntu20.04
RUN pip install torch torchvision torchaudio
RUN pip install transformers
RUN pip install robovlms
```

**요구사항**:
- **CUDA**: 11.8+
- **GPU**: RTX 4090, A100
- **Memory**: 24GB+
- **Storage**: 50GB+

### 7.2 우리 Mobile VLA 배포

**Jetson 최적화**:
```dockerfile
FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install transformers
RUN pip install robovlms
```

**요구사항**:
- **CUDA**: 11.8+ (Jetson)
- **GPU**: Jetson AGX Orin
- **Memory**: 16GB
- **Storage**: 20GB+

## 8. 핵심 차이점 요약

### 8.1 아키텍처 차이

| **구분** | **기존 RoboVLMs** | **우리 Mobile VLA** |
|----------|-------------------|---------------------|
| **액션 공간** | 7-DOF (3D manipulation) | 2D (Mobile navigation) |
| **데이터셋** | CALVIN (24K) | Custom (1K) |
| **환경** | 시뮬레이션 | Real-world |
| **로봇** | Franka Panda | Mobile Robot |

### 8.2 학습 방법 차이

| **구분** | **기존 RoboVLMs** | **우리 Mobile VLA** |
|----------|-------------------|---------------------|
| **Fine-tuning** | Full Fine-tuning | LoRA Fine-tuning |
| **메모리** | 24GB+ | 8-16GB |
| **배치 크기** | 4-8 | 2-4 |
| **학습률** | 2e-5 | 1e-4 |

### 8.3 배포 환경 차이

| **구분** | **기존 RoboVLMs** | **우리 Mobile VLA** |
|----------|-------------------|---------------------|
| **하드웨어** | RTX 4090, A100 | Jetson AGX Orin |
| **메모리** | 24GB+ | 16GB |
| **전력** | 450W+ | 60W |
| **가격** | $2000+ | $2000 |

## 9. 수정된 부분

### 9.1 학습 부분

**변경사항**:
1. **Full Fine-tuning → LoRA Fine-tuning**: 메모리 효율성
2. **7-DOF → 2D 액션**: Mobile robot에 맞는 액션 공간
3. **CALVIN → Custom 데이터셋**: Real-world 데이터 수집

### 9.2 파인튜닝 부분

**변경사항**:
1. **VLM 백본 동결**: 메모리 절약
2. **LoRA 파라미터만 학습**: 효율적 학습
3. **높은 학습률**: LoRA에 적합한 학습률

### 9.3 추론 부분

**변경사항**:
1. **Jetson 최적화**: ARM 아키텍처 최적화
2. **메모리 최적화**: 16GB 제한 내에서 동작
3. **실시간 추론**: Mobile robot에 적합한 속도

## 10. 참고 자료

- `RoboVLMs/robovlms/model/backbone/base_backbone.py`: VLM + LSTM 통합
- `RoboVLMs/robovlms/model/policy_head/base_policy.py`: LSTM 구현
- `RoboVLMs/robovlms/train/base_trainer.py`: 학습 로직
- `RoboVLMs/configs/calvin_finetune/`: CALVIN 설정
- `RoboVLMs/configs/mobile_vla/`: Mobile VLA 설정
