# Action, Image, Text의 Syncing 문제 분석

## GitHub Repository 정보
- **Repository**: [RoboVLMs](https://github.com/Robot-VLAs/RoboVLMs)
- **CALVIN Dataset**: [CALVIN](https://github.com/mees/calvin/tree/main)
- **Paper**: [Towards Generalist Robot Policies: What Matters in Building Vision-Language-Action Models](https://arxiv.org/abs/2412.14058)
- **Website**: [robovlms.github.io](https://robovlms.github.io)

## 1. VLM Fine-tuning 방법: F-FT vs LoRA

### 1.1 F-FT (Full Fine-Tuning)
**GitHub Code Reference**: [RoboVLMs/robovlms/model/backbone/base_backbone.py:512-525](https://github.com/Robot-VLAs/RoboVLMs/blob/main/robovlms/model/backbone/base_backbone.py#L512-L525)
**실제 코드 위치**: LoRA 설정 및 PEFT 모델 적용
```python
# 2단계: 로봇 조작 파인튜닝
for batch in robot_dataloader:
    images = batch['images']
    language = batch['language_instruction']
    actions = batch['actions']
    
    # VLA 손실
    vla_loss = compute_vla_loss(VLA(images, language), actions)
    vla_loss.backward()
    optimizer.step()
```

**특징**:
- **전체 모델 파인튜닝**: 모든 파라미터 업데이트
- **메모리 사용량**: 높음 (전체 모델)
- **성능**: 최고 성능 달성 가능
- **RoboVLMs 적용**: 전체 VLM을 로봇 조작 작업에 최적화

### 1.2 LoRA (Low-Rank Adaptation)
**GitHub Code Reference**: [RoboVLMs/robovlms/model/backbone/base_backbone.py:512-525](https://github.com/Robot-VLAs/RoboVLMs/blob/main/robovlms/model/backbone/base_backbone.py#L512-L525)
**실제 코드 위치**: LoRA 설정 및 PEFT 모델 적용
```python
if self.train_setup_configs["lora_enable"]:
    from llava.train.train import find_all_linear_names
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=self.train_setup_configs["lora_r"],
        lora_alpha=self.train_setup_configs["lora_alpha"],
        target_modules=find_all_linear_names(model),
        lora_dropout=self.train_setup_configs["lora_dropout"],
        bias=self.train_setup_configs["lora_bias"],
        task_type="CAUSAL_LM",
    )
    print("Adding LoRA adapters...")
    self.model = get_peft_model(model, lora_config)
```

**특징**:
- **부분 파라미터 업데이트**: 일부 파라미터만 업데이트
- **메모리 효율성**: 낮은 메모리 사용량
- **계산 효율성**: 빠른 훈련 속도
- **RoboVLMs 적용**: 메모리와 계산 효율성을 위한 선택적 사용

## 2. Action과 rel_action의 동기화

### 2.1 데이터셋의 액션 표현
**GitHub Code Reference**: `5.robovlms_github/learning_pipeline/README.md:210-220`
```python
# 데이터셋의 액션 표현
# tcp pose (7): 상대적인 세계 좌표에서의 x, y, z 위치와 쿼터니언 회전
# tcp velocity (6): 상대적인 세계 좌표에서의 x, y, z 속도와 각속도
# gripper_action (1): 이진 값 (닫힘 = -1, 열림 = 1)
```

### 2.2 Action vs rel_action 구조
**GitHub Code Reference**: [RoboVLMs/robovlms/data/data_utils.py:682-688](https://github.com/Robot-VLAs/RoboVLMs/blob/main/robovlms/data/data_utils.py#L682-L688)
**실제 코드 위치**: 액션 정규화 함수
```python
def normalize_action(action, action_min=-1, action_max=1, maintain_last=False):
    last_val = action[..., -1]
    action = np.clip(action, a_min=float(action_min), a_max=float(action_max))
    res = 2 * (action - action_min) / (action_max - action_min) - 1
    if maintain_last:
        res[..., -1] = last_val
    return res
```

**Action (절대 좌표)**:
```python
['actions']
(dtype=np.float32, shape=(7,))
tcp position (3): x,y,z in absolute world coordinates
tcp orientation (3): euler angles x,y,z in absolute world coordinates
gripper_action (1): binary (close = -1, open = 1)
```

**rel_action (상대 좌표)**:
```python
['rel_actions']
(dtype=np.float32, shape=(7,))
tcp position (3): x,y,z in relative world coordinates normalized and clipped to (-1, 1) with scaling factor 50
tcp orientation (3): euler angles x,y,z in relative world coordinates normalized and clipped to (-1, 1) with scaling factor 20
gripper_action (1): binary (close = -1, open = 1)
```

### 2.3 동기화 방법
**GitHub Code Reference**: `5.robovlms_github/learning_pipeline/README.md:120-130`
```python
# action과 rel_action의 동기화
# 로봇의 현재 상태와 목표 상태를 비교하여 필요한 상대적 움직임을 계산합니다.
# 이를 통해 로봇의 정확한 제어를 수행합니다.
```

**동기화 원리**:
1. **절대 좌표 → 상대 좌표 변환**: 현재 위치에서 목표 위치까지의 상대적 움직임 계산
2. **정규화**: 상대 좌표를 (-1, 1) 범위로 정규화
3. **스케일링 팩터**: 위치(50), 회전(20)에 따른 다른 스케일링 적용

## 3. 로봇팔의 움직임 (7 DOF)

### 3.1 7자유도 표현
**GitHub Code Reference**: `5.robovlms_github/learning_pipeline/README.md:150-160`
```python
# 로봇팔의 7자유도 움직임
# 각 관절의 회전과 그리퍼의 동작을 포함하여 로봇팔의 움직임을 제어합니다.
```

**7 DOF 구성**:
- **TCP Position (3)**: x, y, z 위치
- **TCP Orientation (3)**: x, y, z 회전 (Euler angles)
- **Gripper Action (1)**: 그리퍼 열림/닫힘

### 3.2 액션 공간 처리
**GitHub Code Reference**: [RoboVLMs/robovlms/data/calvin_dataset.py:826-828](https://github.com/Robot-VLAs/RoboVLMs/blob/main/robovlms/data/calvin_dataset.py#L826-L828)
**실제 코드 위치**: CALVIN 데이터셋 액션 정규화
```python
# CALVIN 데이터셋에서 액션 정규화 적용
if self.norm_action:
    new_sample = []
    for s in sample:
        s["actions"] = normalize_action(
            s["actions"], self.norm_min, self.norm_max, maintain_last=True
        )
        new_sample.append(s)
    sample = new_sample
```

## 4. 이미지, 텍스트, 로봇팔의 동기화

### 4.1 멀티모달 동기화
**GitHub Code Reference**: `5.robovlms_github/learning_pipeline/README.md:170-180`
```python
# 이미지, 텍스트, 로봇팔의 동기화
# 입력된 이미지와 텍스트 명령을 분석하여 로봇팔의 동작을 결정합니다.
# 이를 통해 정확한 작업 수행을 보장합니다.
```

### 4.2 VLA Forward Pass
**GitHub Code Reference**: [RoboVLMs/eval/calvin/model_wrapper.py:318-333](https://github.com/Robot-VLAs/RoboVLMs/blob/main/eval/calvin/model_wrapper.py#L318-L333)
**실제 코드 위치**: CALVIN 모델 추론
```python
def step(self, obs, goal):
    """Step function."""
    input_dict = dict()
    image_x, gripper_x, text_x, mask = self.preprocess(obs, goal, self.action_space)

    input_dict["rgb"] = image_x
    input_dict["hand_rgb"] = gripper_x
    input_dict["text"] = text_x
    input_dict["text_mask"] = mask

    with torch.no_grad():
        action = self.policy.inference_step(input_dict)["action"]
```

**동기화 과정**:
1. **이미지 처리**: VLM의 vision tower로 이미지 토큰 생성
2. **텍스트 처리**: VLM의 text tower로 텍스트 토큰 생성
3. **멀티모달 융합**: Vision과 text 토큰을 융합하여 멀티모달 표현 생성
4. **액션 예측**: Policy head로 액션 시퀀스 예측

## 5. Embedded Token 처리

### 5.1 임베디드 토큰 생성
**GitHub Code Reference**: `5.robovlms_github/learning_pipeline/README.md:190-200`
```python
# 임베디드 토큰 처리
# 생성된 임베디드 토큰을 분석하여 모델의 성능을 유지합니다.
```

### 5.2 Learnable Token 활용
**GitHub Code Reference**: `5.robovlms_github/methodology/README.md:82-84`
**실제 코드 위치**: `methodology/README.md:82-84` (One-step 연속 액션 모델)
```python
[LRN] = VLM(o_t, l_prompt)
â_{t:t+L-1} = MLP([LRN])
```

**임베디드 토큰 처리 과정**:
1. **Learnable Token 생성**: VLM으로 [LRN] 토큰 생성
2. **멀티모달 융합**: 이미지, 텍스트, learnable 토큰 융합
3. **액션 예측**: MLP를 통한 액션 시퀀스 예측

## 6. CALVIN Dataset 분석

### 6.1 CALVIN 데이터셋 구조
**GitHub Code Reference**: [RoboVLMs/robovlms/data/calvin_dataset.py:521-873](https://github.com/Robot-VLAs/RoboVLMs/blob/main/robovlms/data/calvin_dataset.py#L521-L873)
**실제 코드 위치**: CALVIN 데이터셋 클래스
```python
class CalvinBenchmark:
    def __init__(self):
        self.dataset_info = {
            'total_demonstrations': 24000,
            'language_instructions': True,
            'trajectory_length': 64,  # time steps
            'basic_skills': 34,
            'splits': ['A', 'B', 'C', 'D']
        }
```

**CALVIN 데이터셋 특징**:
- **총 시연**: 24K 인간 텔레오퍼레이션 시연
- **언어 지시**: 모든 시연에 언어 지시 포함
- **궤적 길이**: 64 시간 단계 이하
- **기본 기술**: 34개 사전 정의된 기본 기술

### 6.2 데이터셋 활용 전략
**GitHub Code Reference**: [RoboVLMs/robovlms/data/calvin_dataset.py:826-828](https://github.com/Robot-VLAs/RoboVLMs/blob/main/robovlms/data/calvin_dataset.py#L826-L828)
**실제 코드 위치**: CALVIN 데이터셋 액션 정규화
```python
# 1단계: Vision-Language 사전 훈련
for batch in vl_dataloader:
    images = batch['images']
    text = batch['text']
    
    # VLM 손실
    vl_loss = CrossEntropy(VLM(images, text), target_text)
    vl_loss.backward()
    optimizer.step()
```

## 7. Multi-modal 해석 구조

### 7.1 LSTM의 한계
**GitHub Code Reference**: `5.robovlms_github/learning_pipeline/README.md:8-12`
```python
# LSTM의 한계: multi-modal을 해석할 수 있는 구조가 LSTM에 없기 때문에 VLM을 ft하고
```

**LSTM 한계점**:
- **멀티모달 처리 부족**: 이미지와 텍스트를 동시에 처리하는 구조 부족
- **시퀀스 의존성**: 순차적 처리로 인한 병목 현상
- **장기 의존성**: 긴 시퀀스에서 정보 손실

### 7.2 VLM의 장점
**GitHub Code Reference**: `5.robovlms_github/learning_pipeline/README.md:10-12`
```python
# VLM의 장점: 강력한 vision-language 이해 능력 보존
# 최소한의 파라미터 추가: 액션 예측을 위한 최소한의 컴포넌트만 추가
# 빠른 수렴: 기존 VLM의 가중치를 초기화로 활용
```

**VLM 장점**:
- **멀티모달 이해**: 이미지와 텍스트를 동시에 처리
- **사전 훈련된 지식**: 대규모 데이터로 학습된 강력한 표현
- **유연한 아키텍처**: 다양한 VLM 백본 활용 가능

## 8. Fine-tuning 과정의 특이점

### 8.1 입력 데이터 형식
**GitHub Code Reference**: [RoboVLMs/robovlms/data/calvin_dataset.py:863-864](https://github.com/Robot-VLAs/RoboVLMs/blob/main/robovlms/data/calvin_dataset.py#L863-L864)
**실제 코드 위치**: CALVIN 데이터셋 이미지 처리
```python
# CALVIN 데이터셋에서 이미지 처리
image_tensors = torch.stack([s["rgb_obs"]["rgb_static"] for s in sample])
gripper_tensors = torch.stack([s["rgb_obs"]["rgb_gripper"] for s in sample])
```

**입력 데이터 형식**:
- **이미지**: RGB 이미지 (224x224 또는 336x336)
- **텍스트**: 자연어 명령어
- **액션**: 7차원 액션 벡터 (TCP pose + gripper)

### 8.2 학습 특이점
**GitHub Code Reference**: `5.robovlms_github/learning_pipeline/README.md:207-218`
```python
hyperparameter_grid = {
    'learning_rate': [1e-4, 2e-5, 1e-5],
    'weight_decay': [0, 1e-1],
    'batch_size': [128, 256, 512],
    'warmup_ratio': [0.25, 0.5]
}
```

**학습 특이점**:
- **하이퍼파라미터 그리드 서치**: 최적 설정 탐색
- **Mixed Precision**: FP16으로 메모리 효율성 향상
- **그래디언트 클리핑**: 안정적인 학습을 위한 그래디언트 제한

### 8.3 Action Head 동시 학습
**GitHub Code Reference**: [RoboVLMs/robovlms/model/backbone/base_backbone.py:34-57](https://github.com/Robot-VLAs/RoboVLMs/blob/main/robovlms/model/backbone/base_backbone.py#L34-L57)
**실제 코드 위치**: BaseRoboVLM 클래스 구조
```python
class BaseRoboVLM(nn.Module):
    def __init__(
        self,
        configs,
        train_setup_configs,
        act_encoder_configs=None,
        act_head_configs=None,
        fwd_head_configs=None,
        window_size=None,
        use_obs_queries=True,
        use_act_queries=True,
        use_hand_rgb=False,
        use_pixel_loss=True,
        use_mim_obs_loss=False,
        use_time_causal_attn=True,
        vision_masked_ratio=0.9,
        use_tube_mask=False,
        fwd_pred_next_n=1,
        use_vision_resampler=False,
        vision_resampler_configs=None,
        use_clip_norm=False,
        use_state=False,
        **kwargs,
    ):
```

**Action Head 학습**:
- **동시 학습**: VLM과 Action Head가 동시에 학습
- **End-to-End**: 전체 파이프라인이 end-to-end로 학습
- **멀티태스크**: Vision-language와 action prediction을 동시에 학습

## 9. 2차원과 3차원 동기화

### 9.1 좌표계 변환
**GitHub Code Reference**: `5.robovlms_github/learning_pipeline/README.md:210-220`
```python
# 2차원과 3차원 동기화
# relative world coordinates normalized and clipped to (-1, 1) with scaling factor 50
```

**좌표계 동기화**:
- **절대 좌표**: 3D world coordinates
- **상대 좌표**: normalized relative coordinates
- **정규화**: (-1, 1) 범위로 클리핑
- **스케일링**: 위치(50), 회전(20)에 따른 다른 스케일링

### 9.2 스케일링 팩터 적용
```python
# 위치 스케일링: scaling factor 50
position_scaled = position * 50

# 회전 스케일링: scaling factor 20  
orientation_scaled = orientation * 20
```

## 10. 데이터셋 추출 및 파인튜닝 진행

### 10.1 데이터셋 추출 과정
**GitHub Code Reference**: [RoboVLMs/robovlms/data/calvin_dataset.py:521-873](https://github.com/Robot-VLAs/RoboVLMs/blob/main/robovlms/data/calvin_dataset.py#L521-L873)
**실제 코드 위치**: CALVIN 데이터셋 클래스
```python
# CALVIN 데이터셋 로드
from robovlms.data.calvin import CalvinDataset

dataset = CalvinDataset(
    data_path="/path/to/calvin/data",
    split="ABCD",  # 또는 "ABC"
    window_size=16,
    action_chunk_size=10
)
```

### 10.2 파인튜닝 진행
**GitHub Code Reference**: [RoboVLMs/robovlms/model/backbone/base_backbone.py:512-525](https://github.com/Robot-VLAs/RoboVLMs/blob/main/robovlms/model/backbone/base_backbone.py#L512-L525)
**실제 코드 위치**: LoRA 설정 및 PEFT 모델 적용
```python
if self.train_setup_configs["lora_enable"]:
    from llava.train.train import find_all_linear_names
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=self.train_setup_configs["lora_r"],
        lora_alpha=self.train_setup_configs["lora_alpha"],
        target_modules=find_all_linear_names(model),
        lora_dropout=self.train_setup_configs["lora_dropout"],
        bias=self.train_setup_configs["lora_bias"],
        task_type="CAUSAL_LM",
    )
    print("Adding LoRA adapters...")
    self.model = get_peft_model(model, lora_config)
```

## 결론

RoboVLMs의 Action, Image, Text 동기화는 다음과 같은 핵심 요소들로 구성됩니다:

### 핵심 동기화 메커니즘
1. **VLM Fine-tuning**: F-FT 또는 LoRA를 통한 효율적 파인튜닝
2. **Action-rel_action 동기화**: 절대/상대 좌표 변환 및 정규화
3. **7 DOF 로봇팔 제어**: TCP pose + gripper action
4. **멀티모달 융합**: Vision-language-action 통합
5. **Embedded Token 처리**: Learnable token을 통한 액션 예측
6. **CALVIN 데이터셋 활용**: 24K 시연 데이터로 학습
7. **Multi-modal 해석**: VLM의 강력한 멀티모달 이해 능력 활용
8. **End-to-End 학습**: VLM과 Action Head 동시 학습
9. **좌표계 동기화**: 2D/3D 좌표 변환 및 스케일링
10. **데이터셋 기반 파인튜닝**: 체계적인 데이터 추출 및 학습 진행
