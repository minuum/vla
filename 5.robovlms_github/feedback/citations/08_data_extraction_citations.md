# 8. Dataset Extraction and Finetuning - Citation Evidence

## 🔍 **GitHub Code Implementation (100% Confirmed)**

### **8.1 CALVIN Dataset Loading**
- **File**: `RoboVLMs/robovlms/data/calvin_dataset.py:521-602`
- **Implementation**: `DiskCalvinDataset` class for loading CALVIN episodes from disk
- **Code**:
```python
class DiskCalvinDataset(BaseCalvinDataset):
    """디스크에서 CALVIN 에피소드를 로드하는 데이터셋"""
    def __init__(
        self,
        image_fn: Callable,           # 이미지 처리 함수
        tokenizer: Callable,          # 토크나이저 함수
        skip_frames: int = 1,         # 프레임 스킵 수
        save_format: str = "npz",      # 저장 형식 (npz/pkl)
        pretrain: bool = False,       # 사전 훈련 여부
        partial_data=False,          # 부분 데이터 사용 여부
        decoder_type="lstm",          # 디코더 타입
        discrete_action=False,        # 이산 액션 사용 여부
        action_tokenizer=None,         # 액션 토크나이저
        model_name="vicuna",          # 모델 이름
        predict_stop_token=True,       # 정지 토큰 예측 여부
        use_mu_law=False,            # μ-law 사용 여부
        mu_val=255,                   # μ-law 값
        n_bin=256,                    # 이산화 빈 수
        min_action=-1,                 # 액션 최소값
        max_action=1,                  # 액션 최대값
        task_type="calvin_action",     # 태스크 타입
        tcp_rel=False,                # TCP 상대 좌표 사용 여부
        few_shot=False,               # Few-shot 학습 여부
        exclude_tasks=[],             # 제외할 태스크 목록
        **kwargs: Any,                # 추가 키워드 인수들
    ):
```

### **8.2 Episode Loading Implementation**
- **File**: `RoboVLMs/robovlms/data/calvin_dataset.py:615-653`
- **Implementation**: `_load_episode` method for loading consecutive frames
- **Code**:
```python
def _load_episode(self, idx: int, window_size: int) -> Dict[str, np.ndarray]:
    """
    디스크에 개별 파일로 저장된 연속 프레임들을 로드하여 에피소드 딕셔너리로 결합
    Args:
        idx: 첫 번째 프레임의 인덱스
        window_size: 샘플링된 에피소드의 길이
    Returns:
        episode: 모달리티 이름을 키로 하는 에피소드가 포함된 numpy 배열들의 딕셔너리
    """
    # 에피소드 시작/끝 인덱스 계산
    start_idx = self.episode_lookup[idx]                    # 에피소드 시작 인덱스
    end_idx = start_idx + window_size + self.act_step - 1    # 에피소드 끝 인덱스
    right_pad = self.right_pad_lookup[idx]                  # 오른쪽 패딩 값
    idx_range = np.arange(start_idx, end_idx)               # 인덱스 범위 생성
    
    # 액션과 이미지 마스크 초기화
    action_mask = np.ones_like(idx_range)                    # 액션 마스크 (모두 1)
    image_mask = np.ones_like(idx_range)                    # 이미지 마스크 (모두 1)
    
    # 패딩 처리
    if right_pad != 0:
        idx_range[right_pad:] = idx_range[right_pad]        # 패딩 부분 인덱스 복제
        action_mask[right_pad:] = 0                         # 패딩 부분 액션 마스크 0
        image_mask[right_pad:] = 0                          # 패딩 부분 이미지 마스크 0

    # 관찰 공간 키들 수집
    keys = list(chain(*self.observation_space.values()))    # 모든 관찰 키들
    keys.remove("language")                                  # 언어 키 제거
    keys.append("scene_obs")                                # 장면 관찰 키 추가
    
    # 각 파일 인덱스에 대해 에피소드 로드
    episodes = [
        self.load_file(self._get_episode_name(file_idx)) for file_idx in idx_range
    ]
    # 키별로 에피소드들 스택
    episode = {key: np.stack([ep[key] for ep in episodes]) for key in keys}
    
    # 언어 데이터 처리
    if self.with_lang:
        episode["language"] = self.lang_ann[self.lang_lookup[idx]]  # 언어 어노테이션 로드
        if self.text_aug:  # 텍스트 증강 사용 시
            task = self.lang_task[self.lang_lookup[idx]]            # 태스크 정보
            enrich_lang = random.choice(                             # 랜덤 언어 선택
                self.enrich_lang[task] + [episode["language"]]
            )
            episode["language"] = enrich_lang                        # 증강된 언어로 교체
    
    # 마스크 정보 추가
    episode["action_mask"] = action_mask    # 액션 마스크 저장
    episode["image_mask"] = image_mask      # 이미지 마스크 저장
    return episode
```

### **8.3 Partial Data Loading**
- **File**: `RoboVLMs/robovlms/data/calvin_dataset.py:1091-1098`
- **Implementation**: `load_partial_traj_data` function for loading specific episodes
- **Code**:
```python
def load_partial_traj_data():
    """부분 궤적 데이터 로드 (data_name_list.txt에서)"""
    # data_name_list.txt 파일 경로 설정
    file = open(
        f"{Path(os.path.abspath(robovlms.__path__[0])).parent.as_posix()}/configs/data/calvin/data_name_list.txt",
        "r",
    )
    lines = file.readlines()                                    # 파일의 모든 줄 읽기
    # 각 줄을 파싱하여 튜플로 변환 (첫 번째 요소 제외하고 정수로 변환)
    lines = [tuple([int(_) for _ in l.split()[1:]]) for l in lines]
    return lines
```

### **8.4 Data Chunking Implementation**
- **File**: `RoboVLMs/robovlms/data/data_utils.py:249-270`
- **Implementation**: `generate_chunck_data` function for creating data chunks
- **Code**:
```python
def generate_chunck_data(data, window_size, chunk_size):
    """데이터 청킹 생성 함수"""
    if data is None:
        return None
    
    bs, seq_len = data.shape[:2]           # 배치 크기, 시퀀스 길이
    raw_data_shape = data.shape[2:]         # 원본 데이터 형태
    data_flatten = data.flatten().view(bs, seq_len, -1)  # 데이터 평탄화
    
    # 시퀀스 길이 검증
    assert (
        seq_len == window_size + chunk_size
    ), f"The sequence length should be {window_size + chunk_size}"
    
    # 윈도우 크기만큼 데이터 반복
    data_flatten = repeat(data_flatten, "b s d -> b w s d", w=window_size)

    # 클로 매트릭스 마스크 생성
    mask = claw_matrix(seq_len, chunk_size - 1, data_flatten.device)
    mask = mask[:window_size].bool()        # 윈도우 크기만큼 마스크 자르기

    # 마스크를 배치 차원으로 확장
    mask = repeat(mask, "w s -> b w s d", b=bs, d=data_flatten.shape[-1])
    data_flatten = torch.masked_select(data_flatten, mask)  # 마스크 적용

    # 최종 데이터 형태로 변환
    data_flatten = data_flatten.view(bs, window_size, chunk_size, *raw_data_shape)
    return data_flatten
```

### **8.5 Training Pipeline**
- **File**: `RoboVLMs/robovlms/train/base_trainer.py:345-395`
- **Implementation**: `_process_batch` method for batch processing
- **Code**:
```python
def _process_batch(self, batch):
    """
    배치 처리 메서드 (다양한 태스크 지원)
    
    Action Prediction:
        args: rgb, language, attention_mask, hand_rgb, action
        reformat: action to input and target (seq_len = window size + chunck size)
    Video Prediction:
        args: rgb, language, attention mask, hand_rgb
        reformat: rgb, [hand_rgb] to input and target (seq_len = window size + chunck size)
    Video Caption:
        args: rgb, language, attention_mask
        reformat: Identity
    Image Caption:
        args: rgb, language, attention_mask
        reformat: Identity
        seq_len = 1
    """
    # 배치가 리스트인 경우 첫 번째 요소 사용
    if isinstance(batch, list):
        batch = batch[0]
    
    # RGB 데이터가 리스트인 경우 GPU로 이동
    if isinstance(batch["rgb"], list):
        rgb = [_.cuda() for _ in batch["rgb"]]
    else:
        rgb = batch["rgb"].cuda()
        if len(rgb.shape) == 4:
            rgb = rgb.unsqueeze(1)
        assert len(rgb.shape) == 5

    if isinstance(batch["text"], list) and isinstance(batch["text"][0], str):
        raise ValueError("The raw text data is not supported")
    else:
        seq_len = self.configs["window_size"]
        language = batch["text"].cuda()
        text_mask = batch["text_mask"].cuda()

    if batch.get("action", None) is not None:
        action = batch["action"].cuda()
    else:
        action = None

    attention_mask = batch.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = batch["attention_mask"].cuda()

    if self.use_hand_rgb and batch.get("hand_rgb", None) is not None:
        hand_rgb = batch["hand_rgb"].cuda()
    else:
        hand_rgb = None

    # Split arm and gripper action
    arm_action = None
    gripper_action = None

    if action is not None:
        arm_action = action[:, :, :6]  # b,len,act_dim-1
        gripper_action = action[:, :, 6]  # b,len
        gripper_action = (gripper_action + 1.0) / 2
        gripper_action = gripper_action.long()

    fwd_rgb_chunck = batch.get("fwd_rgb_chunck", None)
    fwd_hand_rgb_chunck = batch.get("fwd_hand_rgb_chunck", None)
    if fwd_rgb_chunck is not None:
        fwd_rgb_chunck = fwd_rgb_chunck.cuda()
    if fwd_hand_rgb_chunck is not None:
        fwd_hand_rgb_chunck = fwd_hand_rgb_chunck.cuda()

    arm_action_chunck = None
    gripper_action_chunck = None
    action_chunck = batch.get("action_chunck", None)
    if action_chunck is not None:
        action_chunck = action_chunck.cuda()
        arm_action_chunck = action_chunck[..., :6]
        gripper_action_chunck = action_chunck[..., -1]

    if isinstance(rgb, torch.Tensor):
        rgb = rgb[:, :seq_len]
        if hand_rgb is not None:
            hand_rgb = hand_rgb[:, :seq_len]

    chunck_mask = batch.get("chunck_mask", None)
    if chunck_mask is not None:
        chunck_mask = chunck_mask.cuda()

    fwd_mask = batch.get("fwd_mask", None)
    if fwd_mask is not None:
        fwd_mask = fwd_mask.bool().cuda()

    # data preparation for discrete action inputs and labels
    instr_and_action_ids = batch.get("instr_and_action_ids", None)
    if instr_and_action_ids is not None:
```

## 📊 **Data Processing Evidence**

### **8.6 Dataset Preprocessing**
- **Image Processing**: RGB image normalization and augmentation
- **Text Processing**: Language instruction tokenization
- **Action Processing**: 7-DOF action normalization and chunking
- **Window Processing**: Temporal sequence windowing

### **8.7 Training Configuration**
- **Batch Size**: Configurable through training configs
- **Learning Rate**: Configurable through training configs
- **Weight Decay**: Configurable through training configs
- **Warmup Ratio**: Configurable through training configs

### **8.8 Memory Optimization**
- **Mixed Precision**: FP16 for memory efficiency
- **Gradient Checkpointing**: Reduced memory usage
- **Gradient Accumulation**: Effective larger batch sizes
- **LoRA**: Parameter-efficient finetuning

## 🎯 **Key Findings**

1. **Scalable Pipeline**: Handles large-scale CALVIN dataset
2. **Efficient Training**: LoRA-based parameter-efficient finetuning
3. **Memory Optimized**: Mixed precision and gradient checkpointing
4. **Configurable**: Flexible training hyperparameters

## 📁 **Supporting Files**
- `RoboVLMs/robovlms/data/calvin_dataset.py`
- `RoboVLMs/robovlms/data/data_utils.py`
- `RoboVLMs/robovlms/train/base_trainer.py`
- `RoboVLMs/configs/calvin_finetune/*.json` (9 files)
