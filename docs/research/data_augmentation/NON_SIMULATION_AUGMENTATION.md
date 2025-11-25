# Mobile-VLA 비시뮬레이션 데이터 증강 전략

> **목표**: 시뮬레이션 없이 실제 500개 데이터를 5,000개 이상으로 확장

---

## 🎯 전략 개요

시뮬레이션 구축의 높은 비용과 Sim-to-Real Gap을 피하면서도 대규모 데이터 증강을 달성하기 위해, 최신 VLA/VLM 연구에서 검증된 **5가지 비시뮬레이션 증강 방법**을 제안합니다.

---

## 📚 방법 1: CAST - Counterfactual Augmentation (VLM 활용)

### 개요
**CAST (Counterfactual Augmentation with Synthetic Trajectories)**: 기존 궤적에서 VLM을 활용해 "만약 ~했다면?" 식의 대안 액션과 명령어를 생성[1]

### 원리
```python
# 기존 데이터
original_trajectory = {
    'image': [img1, img2, img3, ...],
    'instruction': "사무실로 이동",
    'actions': [(v1, ω1), (v2, ω2), ...]
}

# VLM에 질의
query = "At frame 10, what alternative actions could the robot take?"
vlm_response = "Turn left to avoid obstacle" or "Stop to wait for person"

# 새 데이터 생성
augmented_trajectory = {
    'image': [img1, ..., img10, ...],
    'instruction': "장애물 피해 왼쪽으로 우회",
    'actions': [(v1, ω1), ..., (0.0, +1.5), ...]  # 좌회전 액션
}
```

### Mobile-VLA 적용 방안

#### Step 1: VLM 선택
```python
# GPT-4V 또는 LLaVA 활용
from openai import OpenAI

client = OpenAI()

def generate_counterfactual(image, original_action, timestep):
    prompt = f"""
    This is a mobile robot's view at timestep {timestep}.
    Original action: linear_vel={original_action[0]:.2f}, angular_vel={original_action[1]:.2f}
    
    Suggest 3 alternative valid actions the robot could take and describe why:
    1. (Action, Reason)
    2. (Action, Reason)
    3. (Action, Reason)
    """
    
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}}
            ]
        }]
    )
    
    return parse_alternative_actions(response.choices[0].message.content)
```

#### Step 2: 자동 증강 파이프라인
```python
def cast_augmentation(dataset, samples_per_trajectory=5):
    """
    각 궤적에서 무작위 타임스텝을 샘플링하여 대안 액션 생성
    
    500 trajectories × 5 alternatives = 2,500 new samples
    """
    augmented_data = []
    
    for traj in dataset:
        # 무작위 타임스텝 선택
        keyframes = random.sample(range(len(traj)), k=samples_per_trajectory)
        
        for t in keyframes:
            # VLM으로 대안 생성
            alternatives = generate_counterfactual(
                traj['images'][t],
                traj['actions'][t],
                t
            )
            
            for alt_action, alt_instruction in alternatives:
                # 새 궤적 합성
                new_traj = traj.copy()
                new_traj['actions'][t:] = modify_trajectory(
                    traj['actions'][t:], 
                    alt_action
                )
                new_traj['instruction'] = alt_instruction
                augmented_data.append(new_traj)
    
    return augmented_data
```

### 예상 결과
- **데이터 수**: 500 → 500 + 2,500 = **3,000개**
- **다양성**: 언어 grounding 향상, 다양한 의도 학습
- **품질**: VLM이 물리적으로 타당한 액션만 제안

---

## 📚 방법 2: RESample - Bottleneck States Recovery

### 개요
**RESample (Recovery Exploration Sampling)**: 성공 궤적에서 실패 가능성이 높은 "병목 상태"를 찾아 복구 액션을 학습[2]

### 원리
```
정상 궤적: ────●────●────●──→ Goal
               ↓ (bottleneck: 좁은 통로)
실패 복구:     └──●──●──→ Goal
                  (recovery action)
```

### Mobile-VLA 적용 방안

#### Step 1: Bottleneck 감지
```python
def detect_bottlenecks(trajectory):
    """
    병목 상태 기준:
    1. 높은 각속도 변화 (급회전)
    2. 속도 급감 (장애물 근접)
    3. 반복적인 같은 패턴 (막힌 상태)
    """
    bottlenecks = []
    
    for t in range(len(trajectory) - 1):
        action_t = trajectory['actions'][t]
        action_t1 = trajectory['actions'][t+1]
        
        # 급회전 감지
        angular_change = abs(action_t1[1] - action_t[1])
        if angular_change > 0.5:  # rad/s
            bottlenecks.append(('sharp_turn', t))
        
        # 급정지 감지
        velocity_drop = action_t[0] - action_t1[0]
        if velocity_drop > 0.3:  # m/s
            bottlenecks.append(('sudden_stop', t))
    
    return bottlenecks
```

#### Step 2: 탐색적 복구 액션 생성
```python
def generate_recovery_samples(trajectory, bottleneck_idx):
    """
    병목 지점에서 실패 시나리오와 복구 액션 생성
    """
    recovery_samples = []
    
    # 실패 시나리오: 병목에서 잘못된 액션
    failed_actions = [
        (0.0, 0.0),     # 멈춤
        (0.5, +2.0),    # 과도한 좌회전
        (0.5, -2.0),    # 과도한 우회전
    ]
    
    for failed_action in failed_actions:
        # 실패 궤적 생성
        failed_traj = trajectory.copy()
        failed_traj['actions'][bottleneck_idx] = failed_action
        
        # 복구 액션: 원래 목표로 돌아가기
        recovery_actions = compute_recovery_path(
            failed_position=simulate_action(trajectory['images'][bottleneck_idx], failed_action),
            target_position=trajectory['positions'][bottleneck_idx + 5]
        )
        
        failed_traj['actions'][bottleneck_idx+1:bottleneck_idx+6] = recovery_actions
        failed_traj['label'] = 'recovery'
        
        recovery_samples.append(failed_traj)
    
    return recovery_samples
```

### 예상 결과
- **데이터 수**: 병목 상태 100개 × 3 복구 시나리오 = **300개**
- **효과**: 장애물 회피, 복구 능력 향상
- **강건성**: 실패 상태에서 회복하는 법 학습

---

## 📚 방법 3: ControlNet + Stable Diffusion (이미지 증강)

### 개요
실제 이미지의 구조(depth, edge)를 유지하면서 배경/조명/스타일만 변경하여 시각적 다양성 확보

### Mobile-VLA 적용 방안

#### Step 1: ControlNet 파이프라인 구축
```python
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from transformers import pipeline as hf_pipeline
import torch

# ControlNet 모델 로드
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11f1p_sd15_depth"
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16
).to("cuda")

# Depth Estimator
depth_estimator = hf_pipeline("depth-estimation", model="Intel/dpt-large")

def augment_image_with_controlnet(image, prompt):
    """
    이미지의 깊이 맵을 유지하며 스타일 변경
    """
    # 1. Depth Map 추출
    depth_map = depth_estimator(image)['depth']
    
    # 2. ControlNet으로 이미지 생성
    augmented = pipe(
        prompt=prompt,
        image=depth_map,
        num_inference_steps=20,
        controlnet_conditioning_scale=0.8
    ).images[0]
    
    return augmented
```

#### Step 2: 다양한 스타일 프롬프트
```python
augmentation_prompts = [
    # 조명 변화
    "bright office hallway, fluorescent lighting, professional",
    "dim corridor, evening light, warm atmosphere",
    "natural daylight from windows, sunny day",
    
    # 환경 변화
    "modern glass building interior, reflective surfaces",
    "industrial warehouse, concrete floors",
    "hospital corridor, clean white walls",
    
    # 날씨/시간
    "rainy day, wet floors, gloomy lighting",
    "night scene, artificial lighting, dark shadows",
    
    # 복잡도 변화
    "crowded hallway with people walking",
    "empty corridor, minimal furniture",
    "cluttered office space, many objects"
]

def batch_augment_dataset(dataset, prompts):
    """
    500 images × 10 prompts = 5,000 augmented images
    """
    augmented = []
    
    for img_data in tqdm(dataset):
        original_image = img_data['image']
        
        for prompt in prompts:
            aug_img = augment_image_with_controlnet(original_image, prompt)
            
            # 액션 레이블은 동일 (depth 유지했으므로 물리적으로 유효)
            augmented.append({
                'image': aug_img,
                'action': img_data['action'],
                'instruction': img_data['instruction'],
                'augmentation_type': 'controlnet',
                'prompt': prompt
            })
    
    return augmented
```

### 예상 결과
- **데이터 수**: 500 × 10 = **5,000개**
- **시각 다양성**: 다양한 조명/환경에서 강건성 확보
- **물리적 유효성**: Depth map 유지로 액션 레이블 정확성 보장

---

## 📚 방법 4: Contrastive Learning (Self-Supervised)

### 개요
**CLASP (Contrastive Language-Action-State Pre-training)**: 언어와 로봇 행동을 shared embedding에 정렬하여 적은 데이터로 효율적 학습[3]

### Mobile-VLA 적용 방안

#### Step 1: Contrastive Pre-training
```python
import torch.nn.functional as F

class ContrastiveMobileVLA(nn.Module):
    def __init__(self, vision_encoder, text_encoder, temperature=0.07):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.temperature = temperature
    
    def contrastive_loss(self, vision_features, text_features):
        """
        InfoNCE Loss: 같은 (image, instruction) 쌍은 가깝게, 다른 쌍은 멀게
        """
        # Normalize
        vision_features = F.normalize(vision_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # Cosine similarity
        logits = torch.matmul(vision_features, text_features.T) / self.temperature
        
        # Cross-entropy loss
        labels = torch.arange(len(vision_features)).to(logits.device)
        loss = F.cross_entropy(logits, labels)
        
        return loss
```

#### Step 2: Data Augmentation for Contrastive Learning
```python
def create_contrastive_pairs(dataset):
    """
    하나의 이미지에서 여러 긍정/부정 쌍 생성
    
    긍정 쌍: (image, 올바른 instruction)
    부정 쌍: (image, 다른 instruction)
    
    → 500 images → 2,500 pairs (1 pos + 4 neg per image)
    """
    pairs = []
    
    for i, data in enumerate(dataset):
        image = data['image']
        true_instruction = data['instruction']
        
        # Positive pair
        pairs.append({
            'image': image,
            'instruction': true_instruction,
            'label': 1  # positive
        })
        
        # Negative pairs (random other instructions)
        negative_instructions = random.sample(
            [d['instruction'] for j, d in enumerate(dataset) if j != i],
            k=4
        )
        
        for neg_inst in negative_instructions:
            pairs.append({
                'image': image,
                'instruction': neg_inst,
                'label': 0  # negative
            })
    
    return pairs  # 500 × 5 = 2,500 pairs
```

### 예상 결과
- **데이터 효율성**: 같은 이미지에서 여러 학습 샘플 생성
- **언어 정렬**: Vision-Language alignment 향상
- **Few-shot 성능**: 새로운 명령어에 빠르게 적응

---

## 📚 방법 5: Trajectory Interpolation (궤적 보간)

### 개요
두 성공 궤적 사이를 부드럽게 보간하여 새로운 유효 궤적 생성

### Mobile-VLA 적용 방안

#### Step 1: 궤적 임베딩
```python
def embed_trajectory(trajectory, encoder):
    """
    궤적을 latent space로 인코딩
    """
    image_features = encoder(trajectory['images'])
    action_features = encode_actions(trajectory['actions'])
    
    # Trajectory embedding (평균 pooling)
    traj_embedding = torch.cat([image_features, action_features], dim=-1).mean(dim=0)
    
    return traj_embedding
```

#### Step 2: 보간 및 디코딩
```python
def interpolate_trajectories(traj_A, traj_B, num_samples=5):
    """
    A와 B 사이를 선형 보간
    
    2개 궤적 → 5개 새 궤적
    """
    embed_A = embed_trajectory(traj_A, encoder)
    embed_B = embed_trajectory(traj_B, encoder)
    
    interpolated_trajs = []
    
    for alpha in np.linspace(0.2, 0.8, num_samples):
        # Latent space 보간
        embed_interp = alpha * embed_A + (1 - alpha) * embed_B
        
        # 디코딩 (새 궤적 생성)
        new_traj = decoder(embed_interp)
        
        # 물리적 타당성 검증
        if is_physically_valid(new_traj):
            interpolated_trajs.append(new_traj)
    
    return interpolated_trajs
```

#### Step 3: Pair-wise 증강
```python
# 500 trajectories → choose 100 pairs
trajectory_pairs = random_pairs(dataset, n_pairs=100)

augmented_trajs = []
for traj_A, traj_B in trajectory_pairs:
    augmented_trajs.extend(
        interpolate_trajectories(traj_A, traj_B, num_samples=5)
    )

# 100 pairs × 5 interpolations = 500 new trajectories
```

### 예상 결과
- **데이터 수**: +500개
- **부드러움**: 자연스러운 궤적 생성
- **효율성**: 인코더/디코더만 학습하면 무한 생성 가능

---

## 📊 전체 증강 계획 요약

| 방법 | 생성 데이터 수 | 구축 시간 | 다양성 | 품질 |
|------|--------------|----------|-------|------|
| **CAST (VLM)** | +2,500 | 1주 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **RESample** | +300 | 3일 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **ControlNet** | +5,000 | 1주 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Contrastive** | +2,500 (pairs) | 3일 | ⭐⭐⭐ | ⭐⭐⭐ |
| **Interpolation** | +500 | 1주 | ⭐⭐ | ⭐⭐⭐⭐ |
| **총합** | **+10,800** | **3-4주** | 매우 높음 | 높음 |

### 최종 데이터셋
- **Original**: 500
- **Augmented**: 10,800
- **Total**: **11,300 samples**

---

## 🎯 권장 실행 우선순위

### Phase 1 (Week 1): 빠른 증강
1. ✅ **ControlNet** (구현 간단 + 대량 생성)
   - 500 → 5,500 (10배)
   
### Phase 2 (Week 2): 고품질 증강
2. ✅ **CAST (VLM)** (의미적 다양성)
   - +2,500 (언어 grounding)
   
### Phase 3 (Week 3): 강건성 증강
3. ✅ **RESample** (복구 능력)
   - +300 (실패 복구 학습)

### Phase 4 (Optional): 추가 증강
4. ⭐ **Contrastive Learning** (효율성)
5. ⭐ **Trajectory Interpolation** (부드러움)

---

## 💻 통합 코드 예시

```python
# augmentation_pipeline.py

class MobileVLAAugmentationPipeline:
    def __init__(self, dataset_path):
        self.dataset = load_h5_dataset(dataset_path)
        
        # 각 증강 엔진 초기화
        self.controlnet_engine = ControlNetAugmenter()
        self.cast_engine = CASTAugmenter(vlm_model="gpt-4v")
        self.resample_engine = RESampleAugmenter()
    
    def augment_all(self, output_path):
        """
        전체 증강 파이프라인 실행
        """
        print("🚀 Starting augmentation pipeline...")
        
        # Phase 1: ControlNet
        print("\n[Phase 1] ControlNet Augmentation...")
        controlnet_data = self.controlnet_engine.augment(
            self.dataset, 
            prompts=AUGMENTATION_PROMPTS
        )
        print(f"✅ Generated {len(controlnet_data)} samples")
        
        # Phase 2: CAST
        print("\n[Phase 2] CAST Augmentation...")
        cast_data = self.cast_engine.augment(
            self.dataset,
            samples_per_traj=5
        )
        print(f"✅ Generated {len(cast_data)} samples")
        
        # Phase 3: RESample
        print("\n[Phase 3] RESample Augmentation...")
        resample_data = self.resample_engine.augment(self.dataset)
        print(f"✅ Generated {len(resample_data)} samples")
        
        # 통합
        final_dataset = (
            self.dataset + 
            controlnet_data + 
            cast_data + 
            resample_data
        )
        
        # 저장
        save_h5_dataset(final_dataset, output_path)
        print(f"\n✅ Final dataset: {len(final_dataset)} samples")
        print(f"   Saved to: {output_path}")

# 실행
if __name__ == "__main__":
    pipeline = MobileVLAAugmentationPipeline("data/mobile_vla_500.h5")
    pipeline.augment_all("data/mobile_vla_augmented_10k.h5")
```

---

## 📈 검증 계획

### Ablation Study
| 실험 조건 | 데이터 수 | Val Loss (예상) | 새 환경 성공률 |
|----------|----------|----------------|--------------|
| Baseline | 500 | 0.213 | Baseline |
| +ControlNet | 5,500 | < 0.20 | +10% |
| +CAST | 8,000 | < 0.18 | +15% |
| +RESample | 8,300 | < 0.17 | +20% |
| All Methods | 11,300 | < 0.15 | +25% |

---

## 🔬 참고 논문

1. **CAST**: "Counterfactual Augmentation with Synthetic Trajectories for VLA"
2. **RESample**: "Recovery Exploration for Out-of-Distribution Data in VLA"
3. **CLASP**: "Contrastive Language-Action-State Pre-training"
4. **ControlNet**: "Adding Conditional Control to Text-to-Image Diffusion Models"
5. **AugWM**: "Augmented World Models for Self-Supervised Adaptation"

---

**작성일**: 2025-11-26  
**다음 단계**: Phase 1 (ControlNet) 구현 시작
