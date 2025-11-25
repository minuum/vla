# Mobile-VLA 데이터 형식 명세서 (Data Format Specification)

> **버전**: v1.0  
> **작성일**: 2025-11-26  
> **기준 데이터**: ROS_action/mobile_vla_dataset/*.h5 (468 episodes)

---

## 📋 개요

Mobile-VLA는 **Vision-Language-Action** 모델로, 로봇의 비전 입력(Video)과 언어 명령(Language)을 받아 모바일 로봇의 제어 액션(Action)을 출력합니다.

### 데이터 플로우

```
┌──────────┐     ┌──────────────┐     ┌──────────┐
│  Video   │────▶│              │────▶│  Action  │
│ (Images) │     │  Mobile-VLA  │     │ (Linear, │
└──────────┘     │    Model     │     │ Angular) │
                 │              │     └──────────┘
┌──────────┐     │              │
│ Language │────▶│              │
│(Instruct)│     └──────────────┘
└──────────┘
```

---

## 🎬 Video (입력)

### 형식

**H5 Key**: `images`

```python
shape: (T, H, W, C)
dtype: uint8
range: [0, 255]
```

### 실제 사양 (Mobile-VLA 데이터셋)

| 속성 | 값 | 설명 |
|------|-----|------|
| **시간 차원 (T)** | 18 steps | 에피소드당 프레임 수 (고정) |
| **높이 (H)** | 720 pixels | 세로 해상도 |
| **너비 (W)** | 1280 pixels | 가로 해상도 (16:9 비율) |
| **채널 (C)** | 3 | RGB 컬러 |
| **Dtype** | `uint8` | 0-255 정수 |
| **총 크기** | 49,766,400 elements/episode | ~50MB/에피소드 |

### 예시

```python
import h5py

with h5py.File('episode_xxx.h5', 'r') as f:
    images = f['images'][:]  # Shape: (18, 720, 1280, 3)
    
    # 첫 프레임
    frame_0 = images[0]  # Shape: (720, 1280, 3), uint8
    
    # 특정 타임스텝 범위
    frames = images[5:10]  # Shape: (5, 720, 1280, 3)
```

### 전처리 (모델 입력용)

```python
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("google/paligemma-3b-pt-224")

# 리사이즈 및 정규화
processed_image = processor(
    images=frame_0,  # (720, 1280, 3)
    return_tensors="pt"
)
# Output: (1, 3, 224, 224), float32, [-1, 1]
```

### 주요 특징

- ✅ **고해상도**: 720p (HD) 품질
- ✅ **RGB 순서**: OpenCV BGR이 아닌 RGB
- ✅ **일정 길이**: 모든 에피소드 18 프레임 (간단한 배치 처리)
- ⚠️ **대용량**: 에피소드당 ~50MB

---

## 💬 Language (입력)

### 형식

**H5 Key**: `language_instruction`

```python
shape: (1,)
dtype: 'S256' (bytes, max 256 characters)
encoding: UTF-8
```

### 실제 사양 (Mobile-VLA 데이터셋)

| 속성 | 값 | 설명 |
|------|-----|------|
| **Key** | `language_instruction` | H5 데이터셋 키 |
| **Shape** | (1,) | 에피소드당 1개 명령어 (고정) |
| **Dtype** | `S256` | Bytes string, UTF-8 인코딩 |
| **원본 (한글)** | "장애물을 피해 음료수 페트병 앞으로 도착해라" | 실제 수집 태스크 |
| **영어 번역** | "Navigate around obstacles and reach the front of the beverage bottle" | 기본 명령어 |

### 태스크별 Instruction 변형

파일명에 따라 자동으로 변형된 instruction이 포함되어 있습니다:

| 파일명 패턴 | Instruction |
|-----------|-------------|
| `*_hori_left_*` | "Navigate around obstacles and reach the front of the beverage bottle **on the left**" |
| `*_hori_right_*` | "Navigate around obstacles and reach the front of the beverage bottle **on the right**" |
| 기본 | "Navigate around obstacles and reach the front of the beverage bottle" |

### 예시

```python
import h5py

with h5py.File('episode_xxx.h5', 'r') as f:
    instruction_bytes = f['language_instruction'][0]  # bytes
    instruction = instruction_bytes.decode('utf-8')   # str
    
    print(instruction)
    # Output: "Navigate around obstacles and reach the front of the beverage bottle on the left"
```

### 분포

```
총 468개 에피소드:
├─ "... on the left"  : ~224개 (48%)
├─ "... on the right" : ~244개 (52%)
```

**특징**:
- ✅ **실제 태스크 반영**: 음료수 페트병 도달 태스크
- ✅ **장애물 회피**: "Navigate around obstacles" 포함
- ✅ **방향 정보**: 좌/우 명시



---

## 🎯 Action (출력)

### 형식

**H5 Key**: `actions`

```python
shape: (T, D)
dtype: float32
range: [-1.15, 1.15]
```

### 실제 사양 (Mobile-VLA 데이터셋)

| 속성 | 값 | 설명 |
|------|-----|------|
| **시간 차원 (T)** | 18 steps | Video와 동일 |
| **액션 차원 (D)** | **3** | (linear_x, angular_z, gripper) |
| **Dtype** | `float32` | 32비트 부동소수점 |
| **범위** | [-1.15, 1.15] | 정규화된 속도 |

### 액션 차원 상세

#### Dimension 0: `linear_x` (선속도)

| 속성 | 값 |
|------|-----|
| **의미** | 전진(+) / 후진(-) 선속도 |
| **단위** | m/s |
| **범위** | [0.0, 1.15] (실제는 후진 없음) |
| **평균** | 1.02 m/s |
| **표준편차** | 0.36 m/s |

```python
# 예시 값 해석
0.0   → 정지
0.5   → 중간 속도 전진
1.0   → 빠른 전진
1.15  → 최대 속도
```

#### Dimension 1: `angular_z` (각속도)

| 속성 | 값 |
|------|-----|
| **의미** | 좌회전(+) / 우회전(-) 각속도 |
| **단위** | rad/s |
| **범위** | [-1.15, 1.15] |
| **평균** | 0.32 rad/s (약간 좌회전 편향) |
| **표준편차** | 0.75 rad/s |

```python
# 예시 값 해석
 0.0   → 직진
+0.5   → 완만한 좌회전
+1.15  → 급격한 좌회전
-0.5   → 완만한 우회전
-1.15  → 급격한 우회전
```

#### Dimension 2: `gripper` (그리퍼)

| 속성 | 값 |
|------|-----|
| **의미** | 그리퍼 개폐 (추정) |
| **범위** | **항상 0.0** ❗ |
| **평균** | 0.0 |
| **표준편차** | 0.0 |

**분석**: 
- ❌ 현재 데이터셋에서 **전혀 사용되지 않음**
- 📌 모바일 로봇 내비게이션 태스크에서는 불필요
- 🔧 조작 태스크 추가 시 활용 가능

### 액션 예시

```python
# Sample actions from dataset
actions = [
    [0.00, 0.00, 0.0],  # Step 0: 정지 (에피소드 시작)
    [1.15, 0.00, 0.0],  # Step 1: 최대 속도 직진
    [1.15, 0.00, 0.0],  # Step 2: 최대 속도 직진
    [1.15, 0.00, 0.0],  # Step 3: 최대 속도 직진
    [0.00, 1.15, 0.0],  # Step 4: 제자리 좌회전
    [1.15, 1.15, 0.0],  # Step 5: 전진하며 좌회전
    ...
]
```

### 액션 타입 변환

```python
def classify_action(linear_x, angular_z):
    """액션을 이산 타입으로 분류"""
    LINEAR_THRESHOLD = 0.1
    ANGULAR_THRESHOLD = 0.2
    
    is_moving = abs(linear_x) > LINEAR_THRESHOLD
    is_turning = abs(angular_z) > ANGULAR_THRESHOLD
    
    if not is_moving and not is_turning:
        return 'STOP'
    elif is_moving and not is_turning:
        return 'FORWARD' if linear_x > 0 else 'BACKWARD'
    elif not is_moving and is_turning:
        return 'TURN_LEFT' if angular_z > 0 else 'TURN_RIGHT'
    else:
        # 복합 동작
        direction = 'FORWARD' if linear_x > 0 else 'BACKWARD'
        turn = 'LEFT' if angular_z > 0 else 'RIGHT'
        return f'{direction}_{turn}'

# 예시
classify_action(1.15, 0.0)   → 'FORWARD'
classify_action(1.15, 1.15)  → 'FORWARD_LEFT'
classify_action(0.0, 0.0)    → 'STOP'
```

---

## 📊 추가 메타데이터

### action_event_types

**H5 Key**: `action_event_types`

```python
shape: (T,)
dtype: object (bytes)
values: [b'episode_start', b'start_action', ...]
```

**예시**:
```python
[
    b'episode_start',   # Step 0
    b'start_action',    # Step 1
    b'start_action',    # Step 2
    ...
]
```

**용도**: 
- 에피소드 시작/종료 감지
- 액션 시퀀스 분할
- 디버깅 및 분석

---

## 🔧 데이터 로딩 예시

### 기본 로딩

```python
import h5py
import numpy as np

def load_episode(h5_path):
    """단일 에피소드 로드"""
    with h5py.File(h5_path, 'r') as f:
        data = {
            'images': f['images'][:],          # (18, 720, 1280, 3)
            'actions': f['actions'][:],        # (18, 3)
            'event_types': f['action_event_types'][:]  # (18,)
        }
    return data

episode = load_episode('episode_xxx.h5')
```

### PyTorch Dataset

```python
import torch
from torch.utils.data import Dataset
from pathlib import Path

class MobileVLADataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.h5_files = sorted(Path(data_dir).glob('*.h5'))
        self.transform = transform
    
    def __len__(self):
        return len(self.h5_files)
    
    def __getitem__(self, idx):
        with h5py.File(self.h5_files[idx], 'r') as f:
            images = f['images'][:]    # (18, 720, 1280, 3)
            actions = f['actions'][:]  # (18, 3)
        
        # 전처리
        if self.transform:
            images = self.transform(images)
        
        return {
            'images': torch.from_numpy(images).float(),
            'actions': torch.from_numpy(actions[:, :2]).float()  # Gripper 제외
        }
```

---

## 🎓 모델 Input/Output 사양

### 모델 입력 (추론 시)

```python
# Vision Input
images: torch.Tensor
  shape: (batch, T, 3, H, W)  # 전처리 후
  dtype: torch.float32
  range: [0.0, 1.0] or [-1.0, 1.0]  # 정규화 방식에 따라
  example: (8, 8, 3, 224, 224)  # Batch=8, Window=8, 224x224

# Language Input (추가 예정)
instruction: str or List[str]
  example: "Move the box to the left"
  
# Tokenized
input_ids: torch.Tensor
  shape: (batch, seq_len)
  dtype: torch.long
  example: (8, 64)
```

### 모델 출력

```python
# Predicted Actions
pred_actions: torch.Tensor
  shape: (batch, chunk_size, action_dim)
  dtype: torch.float32
  range: [-1.15, 1.15]
  example: (8, 10, 2)  # Batch=8, Chunk=10, [linear_x, angular_z]

# 실제 적용
action = pred_actions[0, 0, :]  # 첫 배치, 첫 액션
linear_x = action[0].item()    # m/s
angular_z = action[1].item()   # rad/s

# ROS 명령으로 변환
from geometry_msgs.msg import Twist
cmd = Twist()
cmd.linear.x = linear_x
cmd.angular.z = angular_z
```

---

## ⚙️ 하이퍼파라미터 (LoRA 학습 기준)

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| **window_size** | 8 | 입력 이미지 시퀀스 길이 |
| **action_chunk** | 10 | 예측 액션 시퀀스 길이 |
| **action_dim** | 2 | 실제 사용 차원 (linear_x, angular_z) |
| **image_size** | 224 | 전처리 후 크기 (정사각형) |
| **batch_size** | 8-16 | 학습 배치 크기 |

### Window & Chunk 설명

```
Episode Timeline (18 steps):
├─ Window 1 (0-7)   → Predict Actions (0-9)
├─ Window 2 (1-8)   → Predict Actions (1-10)
├─ Window 3 (2-9)   → Predict Actions (2-11)
...

Input:  [img_t, img_t+1, ..., img_t+7]  (8 frames)
Output: [act_t, act_t+1, ..., act_t+9]  (10 actions)
```

---

## 📝 데이터 형식 요약표

| 항목 | Key | Shape | Dtype | Range | 비고 |
|------|-----|-------|-------|-------|------|
| **Video** | `images` | (18, 720, 1280, 3) | uint8 | [0, 255] | RGB, 720p |
| **Language** | `language_instruction` | (1,) | S256 (bytes) | - | ✅ UTF-8, 실제 태스크 |
| **Action - Linear** | `actions[:, 0]` | (18,) | float32 | [0.0, 1.15] | m/s |
| **Action - Angular** | `actions[:, 1]` | (18,) | float32 | [-1.15, 1.15] | rad/s |
| **Action - Gripper** | `actions[:, 2]` | (18,) | float32 | 항상 0.0 | 미사용 |
| **Event Types** | `action_event_types` | (18,) | object | - | 메타데이터 |

---

## 🚨 주요 제약사항 및 이슈

### 1. Language Instruction 부재 ✅ **해결됨**

**상태**: ✅ **468개 파일에 추가 완료**  
**내용**: 
- 실제 태스크: "장애물을 피해 음료수 페트병 앞으로 도착해라"
- 영어 번역: "Navigate around obstacles and reach the front of the beverage bottle"
- 방향/시간대별 변형 포함

**사용법**:
```python
with h5py.File(h5_path, 'r') as f:
    instruction = f['language_instruction'][0].decode('utf-8')
```

### 2. Gripper 차원 미사용 ⚠️

**문제**: `actions[:, 2]`가 항상 0.0  
**영향**: 모델이 3D 액션을 예측하지만 실제로는 2D만 사용  
**해결책**:
```python
# 학습 시 gripper 차원 제거
actions_2d = actions[:, :2]  # (T, 2)
```

### 3. 고정 에피소드 길이 ⚠️

**문제**: 모든 에피소드가 정확히 18 스텝  
**영향**: 
- ✅ 배치 처리 간단
- ❌ 다양한 길이 태스크 학습 불가  
**권장**: 가변 길이 지원 위해 padding/masking 추가

---

## 📚 참고 코드

### 완전한 데이터 로더

```python
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
from typing import Dict, Tuple

class MobileVLAH5Dataset(Dataset):
    """Mobile-VLA H5 데이터셋 로더"""
    
    def __init__(
        self,
        data_dir: str,
        window_size: int = 8,
        action_chunk: int = 10,
        image_size: int = 224,
        use_gripper: bool = False
    ):
        self.h5_files = sorted(Path(data_dir).glob('*.h5'))
        self.window_size = window_size
        self.action_chunk = action_chunk
        self.image_size = image_size
        self.use_gripper = use_gripper
    
    def __len__(self) -> int:
        return len(self.h5_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        with h5py.File(self.h5_files[idx], 'r') as f:
            images = f['images'][:]    # (18, 720, 1280, 3)
            actions = f['actions'][:]  # (18, 3)
        
        # Resize images
        # (실제 환경에서는 transforms 사용)
        images_resized = self._resize_images(images)  # (18, 224, 224, 3)
        
        # Normalize to [0, 1]
        images_norm = images_resized.astype(np.float32) / 255.0
        
        # Action dimension 선택
        if self.use_gripper:
            actions_used = actions  # (18, 3)
        else:
            actions_used = actions[:, :2]  # (18, 2)
        
        return {
            'images': torch.from_numpy(images_norm),  # (18, 224, 224, 3)
            'actions': torch.from_numpy(actions_used).float()  # (18, 2 or 3)
        }
    
    def _resize_images(self, images):
        """이미지 리사이즈 (실제로는 cv2 또는 PIL 사용)"""
        import cv2
        resized = np.zeros((len(images), self.image_size, self.image_size, 3), dtype=np.uint8)
        for i, img in enumerate(images):
            resized[i] = cv2.resize(img, (self.image_size, self.image_size))
        return resized

# 사용 예시
dataset = MobileVLAH5Dataset(
    data_dir='/Users/minu/dev/vla/ROS_action/mobile_vla_dataset',
    window_size=8,
    action_chunk=10,
    use_gripper=False  # Gripper 제외
)

dataloader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4
)

for batch in dataloader:
    images = batch['images']  # (8, 18, 224, 224, 3)
    actions = batch['actions']  # (8, 18, 2)
    break
```

---

**작성**: Mobile-VLA Research Team  
**업데이트**: 2025-11-26  
**다음 단계**: Language instruction 추가 스크립트 작성
