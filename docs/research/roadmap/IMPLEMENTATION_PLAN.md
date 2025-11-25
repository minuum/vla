# Mobile-VLA Implementation Plan

> **목표**: RoboVLMs를 Mobile Robot에 적응시켜 실용적인 VLA 시스템 구축  
> **핵심 질문**: 7DOF Manipulator용 VLM이 2DOF Mobile Robot에 전이 가능한가?

---

## 🎯 핵심 검증 사항

### 1. Context Vector 의미성 검증 (최우선)

**질문**: RoboVLMs가 mobile robot 이미지에서 유의미한 context를 추출하는가?

**접근 방법**:
```python
# 1. Pre-trained RoboVLMs 로드
model = RoboPaligemma.from_pretrained("...")

# 2. Mobile-VLA 이미지 입력
mobile_images = load_samples(n=50)  # 대표 샘플

# 3. Intermediate activation 추출
with model.extract_features() as extractor:
    context_vectors = extractor.forward(mobile_images)

# 4. 분석
- t-SNE 시각화 (Manipulator vs Mobile)
- Cosine similarity 계산
- Activation magnitude 비교
```

**성공 기준**:
- ✅ Context vector가 0이 아님
- ✅ 클러스터링이 의미있게 나뉨 (Left vs Right)
- ✅ Manipulator 데이터와 완전히 다르지 않음

**실패 시 대응**:
- Mobile robot 이미지로 Vision encoder 추가 pre-training
- 다른 VLM backbone 시도 (Flamingo, BLIP 등)

---

### 2. 7DOF → 2DOF 적응 가능성

**질문**: Action head만 교체해서 적은 데이터로 학습 가능한가?

**실험 설계**:

| 실험 | Action Head | 학습 데이터 | 예상 결과 |
|------|------------|----------|----------|
| **Exp 1** | Frozen VLM + New 2DOF head | 50개 | Baseline |
| **Exp 2** | Frozen VLM + Adapter + 2DOF | 50개 | 개선? |
| **Exp 3** | LoRA VLM + 2DOF head | 50개 | 최선? |
| **Exp 4** | Exp 3 + 468개 전체 | 468개 | Upper bound |

**구현**:
```python
class Mobile2DOFHead(nn.Module):
    def __init__(self, context_dim=2048, action_dim=2):
        self.adapter = nn.Linear(context_dim, 512)  # Adapter
        self.action_proj = nn.Linear(512, action_dim * chunk_size)
    
    def forward(self, context_vector):
        # context_vector: (B, 2048) from RoboVLMs
        adapted = self.adapter(context_vector)  # (B, 512)
        actions = self.action_proj(adapted)     # (B, 20)  # 2*10
        return actions.reshape(B, 10, 2)
```

**성공 기준**:
- ✅ 50개로 수렴 가능 (Loss < 0.5)
- ✅ 468개로 Val Loss < 0.2

---

## 📋 Phase별 상세 계획

### Phase 1: RoboVLMs 검증 (Week 1-2)

#### 1.1 환경 구축
```bash
# RoboVLMs 설치
cd RoboVLMs
pip install -e .

# Pre-trained 모델 다운로드
python scripts/download_pretrained.py --model paligemma
```

#### 1.2 Context Vector 추출 스크립트

**파일**: `scripts/research/extract_context_vectors.py`

```python
"""
RoboVLMs context vector 추출 및 분석
"""

def extract_context_vectors(model, images, hook_layer='vision_tower'):
    """
    Args:
        model: RoboPaligemma
        images: (N, 3, 224, 224)
        hook_layer: 'vision_tower' or 'multi_modal_projector'
    
    Returns:
        context_vectors: (N, 2048)
    """
    contexts = []
    
    def hook_fn(module, input, output):
        contexts.append(output.detach().cpu())
    
    # Register hook
    target_layer = getattr(model, hook_layer)
    handle = target_layer.register_forward_hook(hook_fn)
    
    # Forward pass
    with torch.no_grad():
        _ = model(images)
    
    handle.remove()
    return torch.cat(contexts, dim=0)

# 사용
model = load_robopaligemma()
mobile_images = sample_mobile_vla_images(n=50)
contexts = extract_context_vectors(model, mobile_images)

# 분석
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2)
embedded = tsne.fit_transform(contexts.numpy())

# 시각화
plt.scatter(embedded[:, 0], embedded[:, 1], 
            c=labels,  # Left=0, Right=1
            cmap='coolwarm')
plt.savefig('context_vector_tsne.png')
```

#### 1.3 실험 일정

| 날짜 | 작업 | 산출물 |
|------|------|--------|
| Day 1-2 | 환경 구축, 모델 로드 | README.md |
| Day 3-4 | Context vector 추출 | NPY 파일 50개 |
| Day 5-6 | t-SNE, 클러스터링 | PNG 시각화 |
| Day 7 | 보고서 작성 | CONTEXT_VECTOR_REPORT.md |

---

### Phase 2: 데이터 증강 (Week 3-6)

#### 2.1 ControlNet 증강 (Week 3-4)

**목표**: 468 → 4,680 (×10)

**스크립트**: `scripts/augmentation/controlnet_augment.py`

```python
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

# 프롬프트 정의
PROMPTS = [
    "bright office hallway, professional lighting",
    "dim corridor, evening light",
    "rainy day, wet floor",
    "crowded hallway with people",
    "industrial warehouse, concrete",
    "hospital corridor, white walls",
    "night scene, artificial lighting",
    "natural daylight from windows",
    "modern glass building interior",
    "cluttered office space, many objects"
]

# 배치 프로세싱
for h5_file in tqdm(h5_files):
    with h5py.File(h5_file, 'r') as f:
        images = f['images'][:]  # (18, 720, 1280, 3)
        actions = f['actions'][:]
        instruction = f['language_instruction'][0].decode('utf-8')
    
    for i, prompt in enumerate(PROMPTS):
        # Depth map 추출
        depth_maps = estimate_depth(images)
        
        # ControlNet 생성
        augmented_images = controlnet_generate(
            images=images,
            depth_maps=depth_maps,
            prompt=prompt
        )
        
        # 저장
        save_augmented_episode(
            f"{h5_file.stem}_aug{i:02d}.h5",
            augmented_images,
            actions,  # 동일한 액션
            instruction
        )
```

**일정**:
- Week 3: ControlNet 환경 구축, 파이프라인 구현
- Week 4: 468개 증강, 품질 검증

#### 2.2 CAST 증강 (Week 5)

**목표**: 후진 동작 생성 (+500)

```python
# GPT-4V 활용
def generate_backward_scenario(image, original_action):
    prompt = f"""
    This mobile robot is moving forward.
    Describe 3 scenarios where it needs to move BACKWARD:
    1. (Scenario, Action)
    2. (Scenario, Action)
    3. (Scenario, Action)
    """
    
    response = gpt4v(image, prompt)
    
    # Parse response
    scenarios = parse_backward_scenarios(response)
    
    return scenarios

# 후진 데이터 생성
backward_episodes = []
for episode in forward_only_episodes:
    scenarios = generate_backward_scenario(
        episode['images'][0],
        episode['actions'][0]
    )
    
    for scenario in scenarios:
        # 새 에피소드 생성
        new_episode = create_backward_episode(
            base_episode=episode,
            backward_action=scenario['action'],
            instruction=scenario['instruction']
        )
        backward_episodes.append(new_episode)
```

---

### Phase 3: Mobile-VLA 학습 (Week 7-10)

#### 3.1 전체 데이터셋 학습 (Week 7)

**변경사항**:
```yaml
# config/train_mobile_vla.yaml

data:
  train_episodes: 375  # 기존 175 → 375
  val_episodes: 93     # 기존 44 → 93
  dataset_path: "ROS_action/mobile_vla_dataset"

model:
  backbone: "RoboPaligemma"
  lora_enable: true
  lora_r: 32
  lora_alpha: 16
  action_dim: 2  # Linear_x, Angular_z만
  window_size: 8
  action_chunk: 10

training:
  max_epochs: 30
  batch_size: 16
  learning_rate: 1e-4
  early_stopping_patience: 5
```

**예상 결과**:
- Train Loss: 0.134 → 0.10
- Val Loss: 0.213 → 0.17

#### 3.2 증강 데이터 학습 (Week 8-9)

| 실험 | 데이터 | Val Loss (예측) |
|------|--------|----------------|
| Baseline | 375 | 0.17 |
| +ControlNet | 4,680 | 0.14 |
| +CAST | 5,180 | 0.12 |

---

### Phase 4: 추론 시스템 (Week 11-12)

#### 4.1 실시간 추론 루프

```python
class RealtimeVLAController:
    def __init__(self, model, camera, robot):
        self.model = model
        self.camera = camera
        self.robot = robot
        
        self.inference_rate = 0.4  # 400ms
        self.control_rate = 0.02   # 20ms
        self.chunk_size = 10
    
    def run(self, instruction: str):
        while not goal_reached():
            # 1. 카메라 캡처
            image = self.camera.capture()
            
            # 2. VLM 추론 (400ms마다)
            start = time.time()
            action_chunk = self.model.predict(
                image=image,
                instruction=instruction,
                chunk_size=self.chunk_size
            )  # (10, 2)
            inference_time = time.time() - start
            
            # 3. Action chunk 실행 (20ms 간격)
            for action in action_chunk:
                self.robot.set_velocity(
                    linear_x=action[0],
                    angular_z=action[1]
                )
                time.sleep(self.control_rate)
            
            # Wait for next inference cycle
            time.sleep(max(0, self.inference_rate - inference_time))
```

#### 4.2 벤치마크

**측정 지표**:
- **추론 속도**: VLM forward pass 시간
- **제어 정확도**: 목표까지 오차 (cm)
- **성공률**: 10회 시도 중 성공 횟수

---

## ⚠️ 주요 리스크 및 완화 방안

### 리스크 1: Context Vector 무의미

**징후**: t-SNE에서 무작위 분포, 모두 0에 가까움

**완화 방안**:
1. Vision encoder 추가 pre-training (Mobile robot 이미지 1000장)
2. Intermediate layer 시도 (더 앞단 feature)
3. 다른 VLM backbone (Flamingo, BLIP-2)

### 리스크 2: 7DOF → 2DOF 불가능

**징후**: 468개로도 Val Loss > 0.5

**완화 방안**:
1. 시뮬레이션 대량 증강 (10,000개)
2. Pre-training on generic mobile navigation (ImageNet 등)
3. Curriculum learning (간단한 태스크부터)

### 리스크 3: 추론 속도 느림

**징후**: Inference time > 400ms

**완화 방안**:
1. TensorRT 최적화
2. INT8 quantization
3. 작은 backbone (PaliGemma-small)
4. Action chunk 크기 증가 (10 → 20)

---

## 📅 타임라인

```
Week 1-2:  ✅ Context Vector 추출 및 분석
Week 3-4:  🔄 ControlNet 증강 (468 → 4,680)
Week 5:    🔄 CAST 후진 생성 (+500)
Week 6:    🔄 데이터 품질 검증
Week 7:    📚 전체 데이터 학습 (468개)
Week 8-9:  📚 증강 데이터 학습
Week 10:   📚 Ablation study
Week 11-12: 🤖 실시간 추론 시스템
Week 13-14: 📝 Mobile-VLA 선행 연구 조사
Week 15-16: 📝 논문 작성
```

---

**작성**: 2025-11-26  
**다음 마일스톤**: Context Vector 추출 완료 (Week 2)
