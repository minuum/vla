# CALVIN 데이터셋과 데이터 플로우

## 1. CALVIN 데이터셋 개요

### 1.1 데이터셋 구성

**CALVIN (Composing Actions from Language and Vision)**:
- **규모**: 24,000 demonstrations
- **태스크**: 34개 기본 로봇 스킬
- **로봇**: Franka Emika Panda 7-DOF 로봇팔
- **환경**: 시뮬레이션 환경 (MuJoCo)

### 1.2 데이터 구조

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

### 1.3 관측 공간 설정

```python
# CALVIN 관측 공간 설정
obs_config = DictConfig({
    "rgb_obs": ["rgb_static", "rgb_gripper"],    # 이미지
    "state_obs": ["robot_obs"],                  # 로봇 상태 (현재 위치/자세)
    "actions": ["rel_actions"],                  # 상대적 액션 (relative actions)
    "language": ["language"],                    # 텍스트 명령
})
```

## 2. robot_obs (로봇 상태) 상세 구조

### 2.1 15차원 벡터 구성

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

### 2.2 robot_obs 활용

**핵심 사용처**:
- `robot_obs[3:6]` (TCP 자세)를 사용하여 World ↔ TCP frame 변환
- 현재 상태를 알아야 rel_action을 적용 가능
- 로봇의 현재 위치와 자세 정보 제공

## 3. 데이터 전처리 파이프라인

### 3.1 이미지 전처리

```python
# 이미지 전처리 과정
def process_rgb(self, episode, observation_space, transforms, seq_idx=0, window_size=0):
    rgb_obs_keys = observation_space["rgb_obs"]  # ["rgb_static", "rgb_gripper"]
    seq_rgb_obs_dict = {}
    
    for rgb_obs_key in rgb_obs_keys:
        rgb_obs = episode[rgb_obs_key]
        
        # 차원 확장 (단일 환경 관측)
        if len(rgb_obs.shape) != 4:
            rgb_obs = np.expand_dims(rgb_obs, axis=0)
        
        # 윈도우 크기만큼 시퀀스 추출
        if window_size == 0 and seq_idx == 0:
            seq_rgb_obs_ = torch.from_numpy(rgb_obs).byte()
        else:
            seq_rgb_obs_ = torch.from_numpy(
                rgb_obs[seq_idx : seq_idx + window_size]
            ).byte()
        
        # 변환 적용 (Resize, Normalize 등)
        if rgb_obs_key in transforms:
            seq_rgb_obs_ = transforms[rgb_obs_key](seq_rgb_obs_)
        
        seq_rgb_obs_dict[rgb_obs_key] = seq_rgb_obs_
    
    return {"rgb_obs": seq_rgb_obs_dict}
```

### 3.2 언어 전처리

```python
# 언어 전처리 과정
def process_language(self, episode, transforms, with_lang):
    if with_lang:
        return {"lang": episode["language"]}
    else:
        return {"lang": "execute random action."}
```

### 3.3 액션 전처리

```python
# 액션 전처리 과정
def process_actions(self, episode, observation_space, transforms):
    # rel_actions는 이미 상대 액션으로 저장됨
    actions = episode["rel_actions"]
    
    # 정규화 적용
    if self.norm_action:
        actions = normalize_action(actions, self.norm_min, self.norm_max)
    
    return {"actions": actions}
```

## 4. 데이터 로더 설정

### 4.1 CALVIN Dataset 초기화

```python
# CALVIN Dataset 초기화
class CalvinDataset(Dataset):
    def __init__(
        self,
        data_dir: Union[str, Path],
        window_size: int = 8,
        fwd_pred_next_n: int = 10,
        norm_action: bool = True,
        norm_min: float = -0.65,
        norm_max: float = 0.65,
        with_lang: bool = True,
        rgb_pad: int = -1,
        gripper_pad: int = -1,
        **kwargs
    ):
        self.window_size = window_size
        self.fwd_pred_next_n = fwd_pred_next_n
        self.norm_action = norm_action
        self.norm_min = norm_min
        self.norm_max = norm_max
        self.with_lang = with_lang
        self.rgb_pad = rgb_pad
        self.gripper_pad = gripper_pad
        
        # 데이터 디렉토리 설정
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)
        self.abs_datasets_dir = data_dir
```

### 4.2 데이터 샘플링

```python
# __getitem__ 메서드 - 데이터 샘플링
def __getitem__(self, idx: Union[int, Tuple[int, int]], fixed_seed=False) -> Dict:
    """
    시퀀스 데이터를 가져옵니다.
    
    Args:
        idx: 시퀀스의 인덱스
        
    Returns:
        로드된 시퀀스
    """
    head = False
    sequence = self._get_sequences(idx, self.window_size, head=head)
    
    # 이미지 전처리
    new_list = []
    np_rgb = copy.deepcopy(sequence["rgb_obs"]["rgb_static"].numpy())
    for i in range(np_rgb.shape[0]):
        new_list.append(Image.fromarray(np_rgb[i, :, :, :].astype(np.uint8)))
    
    image_tensors = self.image_fn(new_list)
    if self.rgb_pad != -1:
        if self.traj_cons:
            image_tensors = self.rgb_shift.forward_traj(
                image_tensors.unsqueeze(0)
            ).squeeze(0)
        else:
            image_tensors = self.rgb_shift(image_tensors)
    
    sequence["rgb_obs"]["rgb_static"] = image_tensors
    
    # 그리퍼 이미지도 동일하게 처리
    new_list = []
    np_gripper = copy.deepcopy(sequence["rgb_obs"]["rgb_gripper"].numpy())
    for i in range(np_gripper.shape[0]):
        new_list.append(Image.fromarray(np_gripper[i, :, :, :].astype(np.uint8)))
    
    gripper_tensors = self.image_fn(new_list)
    if self.gripper_pad != -1:
        if self.traj_cons:
            gripper_tensors = self.gripper_shift.forward_traj(
                gripper_tensors.unsqueeze(0)
            ).squeeze(0)
        else:
            gripper_tensors = self.gripper_shift(gripper_tensors)
    
    sequence["rgb_obs"]["rgb_gripper"] = gripper_tensors
    
    return sequence
```

## 5. 시퀀스 처리

### 5.1 시퀀스 로딩

```python
# _get_sequences 메서드 - 시퀀스 로딩
def _get_sequences(self, idx: int, window_size: int, head: bool = False) -> Dict:
    """
    window_size 길이의 시퀀스를 로드합니다.
    
    Args:
        idx: 시작 프레임의 인덱스
        window_size: 샘플링할 에피소드 길이
        
    Returns:
        다양한 입력 모달리티와 액션의 텐서 딕셔너리
    """
    episode = self._load_episode(idx, window_size)
    
    # 각 모달리티별 전처리
    seq_state_obs = process_state(episode, self.observation_space, self.transforms, self.proprio_state)
    seq_rgb_obs = self.process_rgb(episode, self.observation_space, self.transforms)
    seq_depth_obs = process_depth(episode, self.observation_space, self.transforms)
    seq_acts = process_actions(episode, self.observation_space, self.transforms)
    seq_lang = self.process_language(episode, self.transforms, self.with_lang)
    
    # 상태 정보 추가
    info = get_state_info_dict(episode)
    info = self._add_language_info(info, idx)
    
    # 모든 정보를 하나의 딕셔너리로 결합
    seq_dict = {
        **seq_state_obs,
        **seq_rgb_obs,
        **seq_depth_obs,
        **seq_acts,
        **info,
        **seq_lang,
    }
    seq_dict["idx"] = idx
    seq_dict["action_mask"] = episode["action_mask"]
    seq_dict["image_mask"] = episode["image_mask"]
    
    return seq_dict
```

### 5.2 시퀀스 패딩

```python
# _pad_sequence 메서드 - 시퀀스 패딩
def _pad_sequence(self, seq: Dict, pad_size: int, head: bool = False) -> Dict:
    """
    마지막 프레임을 반복하여 시퀀스를 패딩합니다.
    
    Args:
        seq: 패딩할 시퀀스
        pad_size: 패딩할 프레임 수
        
    Returns:
        패딩된 시퀀스
    """
    # 로봇 상태 패딩
    seq.update({"robot_obs": self._pad_with_repetition(seq["robot_obs"], pad_size)})
    
    # RGB 관측 패딩
    seq.update({
        "rgb_obs": {
            k: self._pad_with_repetition(v, pad_size, head)
            for k, v in seq["rgb_obs"].items()
        }
    })
    
    # 액션 패딩 (상대 액션의 경우)
    if self.relative_actions:
        if head:
            seq_acts = self._pad_with_zeros(seq["actions"], pad_size, head)
        else:
            # 상대 액션의 경우 마지막 차원(gripper)만 반복
            seq_acts = torch.cat([
                self._pad_with_zeros(seq["actions"][..., :-1], pad_size, head),
                self._pad_with_repetition(seq["actions"][..., -1:], pad_size, head),
            ], dim=-1)
        seq.update({"actions": seq_acts})
    
    return seq
```

## 6. 데이터 증강 (Data Augmentation)

### 6.1 Random Shifts Augmentation

```python
# Random Shifts Augmentation
class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad
    
    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, "replicate")
        
        # 랜덤 시프트 적용
        shift = torch.randint(0, 2 * self.pad + 1, size=(n, 2))
        x = x[:, :, shift[:, 0]:shift[:, 0] + h, shift[:, 1]:shift[:, 1] + w]
        
        return x
```

### 6.2 이미지 변환

```python
# 이미지 변환 설정
transforms = {
    "rgb_static": Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    "rgb_gripper": Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}
```

## 7. 데이터 로더 설정

### 7.1 DataLoader 초기화

```python
# DataLoader 설정
def create_dataloader(dataset, batch_size=8, num_workers=4, shuffle=True):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=True,
        drop_last=True
    )
    return dataloader
```

### 7.2 분산 학습 지원

```python
# 분산 학습을 위한 DistributedSampler
def create_distributed_dataloader(dataset, batch_size=8, num_workers=4, world_size=1, rank=0):
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=sampler,
        pin_memory=True,
        drop_last=True
    )
    return dataloader
```

## 8. 데이터 플로우 요약

### 8.1 전체 데이터 플로우

```
CALVIN Dataset
    ↓
Episode Loading (24K demonstrations)
    ↓
Multi-modal Data Extraction
    ├── RGB Images (rgb_static, rgb_gripper)
    ├── Robot State (robot_obs)
    ├── Actions (rel_actions)
    └── Language (language)
    ↓
Data Preprocessing
    ├── Image: Resize, Normalize, Augmentation
    ├── Action: Normalize [-1, 1]
    └── Language: Tokenization
    ↓
Sequence Sampling (window_size=8)
    ↓
Batch Creation (batch_size=8)
    ↓
Model Input
```

### 8.2 핵심 데이터 특성

- **이미지**: 2개 카메라 (정적, 그리퍼) × 224×224×3
- **로봇 상태**: 15차원 벡터 (TCP pose + joint angles + gripper)
- **액션**: 7차원 상대 액션 (translation + rotation + gripper)
- **언어**: 자연어 명령어
- **시퀀스**: 8프레임 윈도우 크기

## 9. 참고 자료

- `RoboVLMs/robovlms/data/calvin_dataset.py`: CALVIN 데이터 로더
- `RoboVLMs/robovlms/data/data_utils.py`: 데이터 유틸리티 함수
- `RoboVLMs/configs/data/calvin/`: CALVIN 데이터 설정
- CALVIN 논문: "CALVIN: A Benchmark for Multimodal Language-Conditioned Imitation Learning for Long-Horizon Robot Manipulation Tasks"
