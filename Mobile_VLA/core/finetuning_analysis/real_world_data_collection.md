# Real-World 데이터 수집 방법

## 1. RoboVLMs가 사용하는 Real-World 데이터

### 1.1 CALVIN 데이터셋 (Real-World 특성)

```python
# CALVIN 데이터셋의 실제 로봇 환경
obs_config = DictConfig({                       # OmegaConf DictConfig (설정 관리)
    "rgb_obs": ["rgb_static", "rgb_gripper"],   # RGB 관측: 정적 카메라 + 그리퍼 카메라
    "depth_obs": [],                           # 깊이 정보 (사용 안함, RGB만 사용)
    "state_obs": ["robot_obs"],                 # 로봇 상태 정보 (7-DOF 관절 각도, 속도)
    "actions": ["rel_actions"],               # 상대적 액션 (TCP frame 기준)
    "language": ["language"]                   # 언어 명령 (태스크 설명)
})
```

**출처**: `RoboVLMs/robovlms/data/calvin_dataset.py:63-71`

**각 설정 상세 설명**:
- **DictConfig**: OmegaConf 설정 객체 (타입 안전성, 중첩 접근)
- **"rgb_obs"**: RGB 이미지 관측 (2개 카메라)
    - **"rgb_static"**: 정적 카메라 (전체 작업 공간 시야)
    - **"rgb_gripper"**: 그리퍼 카메라 (로봇 팔 끝단 시야)
- **"depth_obs"**: 깊이 정보 (빈 리스트, 사용하지 않음)
- **"state_obs"**: 로봇 상태 정보
    - **"robot_obs"**: 15차원 벡터 (7개 관절 각도 + 7개 관절 속도 + 1개 그리퍼 상태)
- **"actions"**: 액션 정보
    - **"rel_actions"**: 상대적 액션 (TCP frame 기준, 일반화 성능 향상)
- **"language"**: 언어 명령
    - **"language"**: 자연어 태스크 설명 (예: "pick up the red block")

### 1.2 Real-World 데이터 구성

**CALVIN supports a range of sensors commonly utilized for visuomotor control**:

1. **Static camera RGB images** - with shape `200x200x3`
2. **Static camera Depth maps** - with shape `200x200`
3. **Gripper camera RGB images** - with shape `84x84x3`
4. **Gripper camera Depth maps** - with shape `84x84`
5. **Tactile image** - with shape `120x160x6`
6. **Proprioceptive state** - EE position (3), EE orientation in euler angles (3), gripper width (1), joint positions (7), gripper action (1)

**Real-World 데이터 구성**:
- **Franka Emika Panda 7-DOF 로봇팔**: 실제 로봇 하드웨어
- **다중 카메라 시스템**: 정적 카메라 + 그리퍼 카메라
- **실제 물리 환경**: 테이블, 물체, 조작 공간
- **다양한 태스크**: pick-and-place, navigation, manipulation
- **실제 로봇 조작**: 전문가가 직접 조작하여 데이터 수집

## 2. Custom Dataset 지원

### 2.1 Custom Dataset 형식

```python
"rgb": image_tensors,           # 정적 카메라 이미지 [Batch, Window, C, H, W]
"hand_rgb": gripper_tensors,    # 그리퍼 카메라 이미지 [Batch, Window, C, H, W]
"action": action_tensors,       # 액션 시퀀스 [Batch, Window, Action_Dim]
"text": text_tensors,           # 토큰화된 텍스트 [Batch, Max_Text_Len]
"text_mask": attention_mask,    # 텍스트 어텐션 마스크 [Batch, Max_Text_Len]
"action_chunk": action_chunk,   # 액션 청크 [Batch, Window, Chunk_Size, Action_Dim]
"chunk_mask": action_mask,      # 유효한 액션 청크 마스크 [Batch, Window, Chunk_Size]
"instr_and_action_ids": instr_and_action_ids,      # 자동회귀 입력 토큰 ID
"instr_and_action_labels": instr_and_action_labels, # 자동회귀 라벨 토큰 ID
"instr_and_action_mask": instr_and_action_mask,     # 자동회귀 마스크
"raw_text": raw_text,           # 원본 자연어 명령 리스트
"data_source": data_source      # 태스크 타입 문자열 (calvin_action 등)
```

**출처**: `RoboVLMs/README.md:300-312`

**각 필드 상세 설명**:
- **"rgb"**: 정적 카메라 이미지 (전체 작업 공간 시야)
    - **Shape**: [Batch_Size, Window_Size, Channel, Height, Width]
    - **용도**: 환경 인식, 물체 위치 파악
- **"hand_rgb"**: 그리퍼 카메라 이미지 (로봇 팔 끝단 시야)
    - **Shape**: [Batch_Size, Window_Size, Channel, Height, Width]
    - **용도**: 그리퍼 근처 물체 세부 인식
- **"action"**: 액션 시퀀스 (7-DOF 로봇 액션)
    - **Shape**: [Batch_Size, Window_Size, Action_Dim] (Action_Dim=7)
    - **용도**: 로봇 제어 명령 (Translation 3 + Rotation 3 + Gripper 1)
- **"text"**: 토큰화된 텍스트 (자연어 → 토큰 ID)
    - **Shape**: [Batch_Size, Max_Text_Len]
    - **용도**: 언어 명령 이해
- **"text_mask"**: 텍스트 어텐션 마스크 (패딩 토큰 무시)
    - **Shape**: [Batch_Size, Max_Text_Len]
    - **용도**: 유효한 토큰만 어텐션 계산
- **"action_chunk"**: 액션 청크 (미래 액션 예측)
    - **Shape**: [Batch_Size, Window_Size, Chunk_Size, Action_Dim]
    - **용도**: 연속된 액션 시퀀스 예측
- **"chunk_mask"**: 액션 청크 마스크 (유효한 청크 표시)
    - **Shape**: [Batch_Size, Window_Size, Chunk_Size]
    - **용도**: 유효한 액션 청크만 학습
- **"instr_and_action_ids"**: 자동회귀 입력 토큰 ID
    - **용도**: VLM의 next-token prediction 학습
- **"instr_and_action_labels"**: 자동회귀 라벨 토큰 ID
    - **용도**: Cross-entropy loss 계산
- **"instr_and_action_mask"**: 자동회귀 마스크
    - **용도**: 유효한 토큰만 loss 계산
- **"raw_text"**: 원본 자연어 명령
    - **용도**: 디버깅, 로깅, 사용자 인터페이스
- **"data_source"**: 태스크 타입 식별자
    - **예시**: "calvin_action", "oxe_action" 등
    - **용도**: 데이터셋 구분, 태스크별 처리

### 2.2 Custom Dataset 설정 예시

**Custom Dataset Config**:

```python
"train_dataset": {                    # 학습 데이터셋 설정
    "type": "CustomDataset",          # 커스텀 데이터셋 클래스
    "data_dir": "path/to/custom_data", # 데이터 디렉토리 경로
    "shift_first": false,             # 첫 번째 프레임 시프트 비활성화
    "model_name": "kosmos",           # VLM 모델명 (토크나이저 매칭)
    "rgb_pad": 10,                    # RGB 이미지 랜덤 시프트 크기
    "gripper_pad": 4                  # 그리퍼 이미지 랜덤 시프트 크기
},
"val_dataset": {                      # 검증 데이터셋 설정
    "type": "CustomDataset",          # 동일한 커스텀 데이터셋 클래스
    "data_dir": "path/to/custom_data", # 동일한 데이터 디렉토리
    "model_name": "kosmos"            # 동일한 VLM 모델명
}
```

**출처**: `RoboVLMs/README.md:368-382`

**각 설정 상세 설명**:
- **"train_dataset"**: 학습용 데이터셋 설정
    - **"type"**: 데이터셋 클래스명 (CustomDataset)
    - **"data_dir"**: 데이터 파일들이 저장된 디렉토리 경로
    - **"shift_first"**: 첫 번째 프레임 시프트 여부 (False=시프트 안함)
    - **"model_name"**: VLM 모델명 (토크나이저 매칭용)
    - **"rgb_pad"**: RGB 이미지 랜덤 시프트 픽셀 수 (데이터 증강)
    - **"gripper_pad"**: 그리퍼 이미지 랜덤 시프트 픽셀 수 (데이터 증강)
- **"val_dataset"**: 검증용 데이터셋 설정
    - **"type"**: 동일한 커스텀 데이터셋 클래스 사용
    - **"data_dir"**: 동일한 데이터 디렉토리 (학습/검증 분할)
    - **"model_name"**: 동일한 VLM 모델명 (일관성 유지)

### 2.3 OpenVLA 데이터셋 설정 (실제 경로)

```python
"train_dataset": {
    "type": "OpenVLADataset",                      # OpenVLA 데이터셋 클래스
    "data_root_dir": "openvla/datasets/open-x-embodiment", # OpenVLA 데이터 루트 경로
    "model_name": "kosmos",                       # Kosmos VLM 모델명
    "image_aug": true,                            # 이미지 증강 활성화
    "mode": "train",                              # 학습 모드
    "data_mix": "bridge",                          # Bridge 데이터 믹스 사용
    "window_sample": "sliding",                    # 슬라이딩 윈도우 샘플링
    "organize_type": "interleave",                 # 인터리브 데이터 구성
    "shuffle_buffer_size": 51200,                  # 셔플 버퍼 크기
    "train": true                                 # 학습 데이터셋 플래그
},
"val_dataset": {
    "type": "OpenVLADataset",                     # 동일한 OpenVLA 데이터셋 클래스
    "data_root_dir": "openvla/datasets/open-x-embodiment", # 동일한 데이터 루트 경로
    "model_name": "kosmos",                       # 동일한 Kosmos 모델명
    "mode": "train",                              # 검증 모드
    "data_mix": "bridge",                          # 동일한 Bridge 데이터 믹스
    "window_sample": "sliding",                   # 동일한 슬라이딩 윈도우
    "organize_type": "interleave",                # 동일한 인터리브 구성
    "shuffle_buffer_size": 10000,                 # 검증용 셔플 버퍼 크기
    "train": false                                # 검증 데이터셋 플래그
}
```

## 3. 데이터 수집 방법

### 3.1 하드웨어 요구사항

**로봇 하드웨어**:
- **7-DOF 로봇팔**: Franka Emika Panda 또는 유사한 로봇
- **다중 카메라 시스템**: 정적 카메라 + 그리퍼 카메라
- **실제 물리 환경**: 테이블, 물체, 조작 공간

**카메라 시스템**:
- **정적 카메라 (rgb_static)**: 전체 작업 공간 시야 (200x200x3)
- **그리퍼 카메라 (rgb_gripper)**: 로봇 팔 끝단 시야 (84x84x3)

### 3.2 데이터 수집 프로세스

**1. 환경 설정**:
```python
# 로봇 환경 설정
robot_config = {
    "robot_type": "franka_panda",      # 로봇 타입
    "camera_positions": ["static", "gripper"],  # 카메라 위치
    "workspace_bounds": [-0.5, 0.5, -0.5, 0.5, 0.0, 0.3],  # 작업 공간 경계
    "sampling_rate": 30                # 샘플링 주파수 (Hz)
}
```

**2. 데이터 수집**:
```python
# 데이터 수집 루프
def collect_episode():
    episode_data = {
        "images": [],      # 이미지 시퀀스
        "actions": [],     # 액션 시퀀스
        "states": [],      # 로봇 상태 시퀀스
        "language": ""     # 언어 명령
    }
    
    # 에피소드 시작
    for timestep in range(episode_length):
        # 이미지 캡처
        static_img = capture_static_camera()
        gripper_img = capture_gripper_camera()
        
        # 로봇 상태 읽기
        robot_state = get_robot_state()
        
        # 액션 실행 (전문가 조작)
        action = expert_control()
        
        # 데이터 저장
        episode_data["images"].append([static_img, gripper_img])
        episode_data["actions"].append(action)
        episode_data["states"].append(robot_state)
    
    # 언어 명령 추가
    episode_data["language"] = input("Enter task description: ")
    
    return episode_data
```

**3. 데이터 저장**:
```python
# HDF5 형식으로 저장
import h5py

def save_episode(episode_data, episode_id):
    with h5py.File(f"episode_{episode_id}.h5", "w") as f:
        f.create_dataset("images", data=episode_data["images"])
        f.create_dataset("actions", data=episode_data["actions"])
        f.create_dataset("states", data=episode_data["states"])
        f.attrs["language"] = episode_data["language"]
        f.attrs["episode_length"] = len(episode_data["actions"])
```

### 3.3 데이터 전처리

**이미지 전처리**:
```python
# 비전 데이터 정규화 (CLIP 표준)
image_mean = [0.48145466, 0.4578275, 0.40821073]  # CLIP ImageNet 정규화 평균
image_std = [0.26862954, 0.26130258, 0.27577711]   # CLIP ImageNet 정규화 표준편차

# 리사이징 및 증강
transforms = [
    Resize((224, 224)),                    # CLIP 입력 크기로 리사이징
    RandomHorizontalFlip(p=0.1),          # 제한적 수평 뒤집기 (10% 확률)
    ColorJitter(brightness=0.1, contrast=0.1),  # 색상 증강 (밝기, 대비)
    Normalize(mean=image_mean, std=image_std)    # CLIP 표준 정규화
]
```

**액션 정규화**:
```python
# 액션 정규화 [-0.65, 0.65] 범위 (CALVIN 표준)
norm_min, norm_max = -0.65, 0.65          # 정규화 범위 (안전한 액션 범위)

# arm 액션 (6-DOF: Translation 3 + Rotation 3)
arm_action = action[:, :, :6]             # 액션의 처음 6차원 (위치 + 회전)
normalized_arm = np.clip(arm_action, norm_min, norm_max)  # 범위 제한

# gripper 액션 (1-DOF): [-1, 1] → [0, 1] 변환 (이진 분류용)
gripper_action = (action[:, :, 6] + 1.0) / 2.0  # 그리퍼 액션 정규화
```

## 4. 데이터 품질 관리

### 4.1 데이터 검증

```python
def validate_episode(episode_data):
    """에피소드 데이터 검증"""
    # 이미지 검증
    assert len(episode_data["images"]) > 0, "No images found"
    assert all(img.shape == (2, 224, 224, 3) for img in episode_data["images"]), "Invalid image shape"
    
    # 액션 검증
    assert len(episode_data["actions"]) > 0, "No actions found"
    assert all(action.shape == (7,) for action in episode_data["actions"]), "Invalid action shape"
    
    # 상태 검증
    assert len(episode_data["states"]) > 0, "No states found"
    assert all(state.shape == (15,) for state in episode_data["states"]), "Invalid state shape"
    
    # 언어 명령 검증
    assert episode_data["language"] != "", "No language command found"
    
    return True
```

### 4.2 데이터 증강

```python
def augment_episode(episode_data):
    """에피소드 데이터 증강"""
    augmented_episodes = []
    
    # 원본 에피소드
    augmented_episodes.append(episode_data)
    
    # 이미지 회전 증강
    for angle in [90, 180, 270]:
        rotated_episode = rotate_images(episode_data, angle)
        augmented_episodes.append(rotated_episode)
    
    # 색상 증강
    color_augmented = color_jitter(episode_data)
    augmented_episodes.append(color_augmented)
    
    return augmented_episodes
```

## 5. 핵심 결론

### 5.1 Real-World 데이터 수집 요구사항

1. **하드웨어**: 7-DOF 로봇팔 + 다중 카메라 시스템
2. **환경**: 실제 물리 환경 (테이블, 물체, 조작 공간)
3. **데이터 형식**: HDF5/JSON 형태로 저장
4. **언어 명령**: 각 에피소드마다 태스크 설명

### 5.2 데이터 품질 관리

1. **검증**: 이미지, 액션, 상태, 언어 명령 검증
2. **전처리**: CLIP 표준 정규화, 액션 정규화
3. **증강**: 이미지 회전, 색상 조정 등

### 5.3 Custom Dataset 지원

1. **유연한 형식**: 다양한 데이터 소스 지원
2. **표준화된 인터페이스**: RoboVLMs와 호환
3. **확장성**: 새로운 태스크에 쉽게 적용 가능
