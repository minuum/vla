# 15-2. VLM Fine-tuning과 LSTM Layer 학습: 학습/추론 변수 및 벤치마크

## 📋 개요

이 문서는 RoboVLMs에서 학습 변수와 추론 변수를 상세히 분석하고, 실제 로봇 벤치마크 데이터셋을 설명합니다.

## 🔧 1. 학습 변수와 추론 변수 상세 분석

### 1.1 학습 변수 (Training Variables)

**BaseRoboVLM._trainable_params_setup() - 학습 가능한 파라미터 설정**
```python
def _trainable_params_setup(self):
    model = self.model  # 백본 VLM 모델 (PaliGemma, Kosmos, LLaVA 등)
    
    # 1단계: 백본 모델 동결 설정
    if self.train_setup_configs["freeze_backbone"]:
        model.requires_grad_(False)  # 전체 모델 동결
    else:
        if self.train_setup_configs.get("train_decoder_layers", -1) == -1:
            model.requires_grad_(True)  # 전체 모델 학습
        else:
            # 마지막 N개 레이어만 학습
            model.requires_grad_(False)
            for layer in self.text_tower.layers[-self.train_setup_configs["train_decoder_layers"]:]:
                layer.requires_grad_(True)
    
    # 2단계: 비전 인코더 동결 설정
    # vision_tower: VLM의 비전 인코더 (CLIP, SigLIP 등)
    if self.train_setup_configs.get("train_vision", False):
        self.vision_tower.requires_grad_(True)
    else:
        self.vision_tower.requires_grad_(False)
    
    # 3단계: LoRA 설정
    if self.train_setup_configs["lora_enable"]:
        # LoRA 파라미터만 학습 가능하도록 설정
        pass
```

**vision_tower와 text_tower 설명**:
- **`vision_tower`**: VLM의 비전 인코더 부분 (이미지 → 특징 벡터)
  - PaliGemma: `model.vision_tower` (SigLIP 기반)
  - Kosmos: `model.vision_model` (CLIP 기반)
  - LLaVA: `model.get_vision_tower()` (CLIP 기반)
  - Flamingo: `self.vision_encoder` (CLIP 기반)

- **`text_tower`**: VLM의 텍스트/언어 모델 부분 (텍스트 → 특징 벡터)
  - PaliGemma: `model.language_model.model` (Gemma Decoder)
  - Kosmos: `model.text_model.model` (Decoder-only Transformer)
  - LLaVA: `model.transformer` (GPT-style Transformer)
  - Flamingo: `self.model` (언어 모델 전체)

**백본별 구현 예시**:
```python
# RoboPaligemma (robopaligemma.py:19-24)
@property
def text_tower(self):
    return self.model.language_model.model  # Gemma Decoder

@property
def vision_tower(self):
    return self.model.vision_tower  # SigLIP

# RoboKosMos (robokosmos.py:16-21)
@property
def text_tower(self):
    return self.model.text_model.model  # Transformer Decoder

@property
def vision_tower(self):
    return self.model.vision_model  # CLIP Vision

# RoboLLaVA (robollava.py:19-24)
@property
def text_tower(self):
    return self.model.transformer  # GPT Transformer

@property
def vision_tower(self):
    return self.model.get_vision_tower()  # CLIP Vision
```

**Kosmos2Processor 공식 문서 근거**:

Hugging Face 공식 문서에 따르면, Kosmos-2는 다음과 같이 구성됩니다:

```python
class transformers.Kosmos2Processor(
    image_processor,  # CLIPImageProcessor
    tokenizer,        # XLMRobertaTokenizerFast
    num_patch_index_tokens = 1024,
    **kwargs
)
```

**Parameters**:
- **image_processor** (`CLIPImageProcessor`) — An instance of `CLIPImageProcessor`. The image processor is a required input.
- **tokenizer** (`XLMRobertaTokenizerFast`) — An instance of `['XLMRobertaTokenizerFast']`. The tokenizer is a required input.

> "Constructs an KOSMOS-2 processor which wraps a KOSMOS-2 image processor and a KOSMOS-2 tokenizer into a single processor."

> "Kosmos2Processor offers all the functionalities of **CLIPImageProcessor** and some functionalities of **XLMRobertaTokenizerFast**."

이것이 Kosmos-2의 `vision_tower`가 CLIP 기반이고, `text_tower`가 XLM-Roberta 기반 Transformer인 이유입니다.

**출처**: 
- [Hugging Face KOSMOS-2 Documentation](https://huggingface.co/docs/transformers/en/model_doc/kosmos-2)
- `RoboVLMs/robovlms/model/backbone/base_backbone.py:470-512`
- `RoboVLMs/robovlms/model/backbone/robopaligemma.py:19-24`
- `RoboVLMs/robovlms/model/backbone/robokosmos.py:16-21`
- `RoboVLMs/robovlms/model/backbone/robollava.py:19-24`
- `RoboVLMs/robovlms/model/backbone/roboflamingo.py:35-40`

**BaseTrainer.get_grouped_params() - 학습 파라미터 그룹화**
```python
def get_grouped_params(self, model):
    return [
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad],
            "weight_decay": self.configs["weight_decay"],
        }
    ]
```

**출처**: `RoboVLMs/robovlms/train/base_trainer.py:716-722`

**RoboFlamingo._trainable_params_setup() - Flamingo 모델 학습 설정**
```python
def _trainable_params_setup(self):
    self.requires_grad_(False)
    
    # 1단계: 비전 인코더 학습 설정
    if self.train_setup_configs["train_vision"]:
        self.vision_encoder.requires_grad_(True)
    
    # 2단계: 디코더 레이어 학습 설정
    if self.train_setup_configs["train_decoder_layers"] == -1:
        self.model.gated_cross_attn_layers.requires_grad_(True)
    else:
        # 마지막 N개 레이어만 학습
        ix = self.train_setup_configs["train_decoder_layers"]
        for layer in self.model.gated_cross_attn_layers[-ix:]:
            layer.requires_grad_(True)
    
    # 3단계: 전체 디코더 학습 설정
    if self.train_setup_configs["train_full_decoder"]:
        self.model.requires_grad_(True)
    
    # 4단계: 리샘플러 학습 설정
    if self.train_setup_configs["train_resampler"]:
        self.perceiver.requires_grad_(True)
    else:
        self.perceiver.requires_grad_(False)
    
    # 5단계: 텍스트 임베딩 학습 설정
    if self.train_setup_configs["train_text_embedding"]:
        self.model.get_input_embeddings().requires_grad_(True)
    else:
        self.model.get_input_embeddings().requires_grad_(False)
    
    # 6단계: 액션 헤드 학습 설정
    self.act_head.requires_grad_(True)
```

**출처**: `RoboVLMs/robovlms/model/backbone/roboflamingo.py:131-156`

### 1.2 추론 변수 (Inference Variables)

**BaseRoboVLM.inference() - 추론 모드 설정**
```python
def inference(
    self,
    vision_x: torch.Tensor,
    lang_x: torch.Tensor,
    attention_mask: torch.Tensor = None,
    position_ids: torch.LongTensor = None,
    use_cached_vision_x: bool = False,
    action_labels: Tuple[torch.Tensor, torch.Tensor] = None,
    action_mask: torch.Tensor = None,
    caption_labels: torch.Tensor = None,
    caption_mask: torch.Tensor = None,
    past_key_values=None,
    use_cache: bool = False,
    vision_gripper=None,
    **kwargs,
):
    prediction = {}
    
    # 1단계: 입력 검증
    assert vision_x is not None
    bs, seq_len = vision_x.shape[:2]
    action_space = self.act_head_configs.get("action_space", "continuous")
    
    # 2단계: 액션 예측
    if self.train_setup_configs["predict_action"]:
        if action_space == "discrete":
            action = self.pred_action_discrete(
                lang_x, vision_x, vision_gripper, attention_mask
            )
            prediction["action"] = action
        else:
            prediction["action"] = self.forward_continuous(
                vision_x,
                lang_x,
                attention_mask,
                vision_gripper=vision_gripper,
                mode="inference",
            )
    
    return prediction
```

**출처**: `RoboVLMs/robovlms/model/backbone/base_backbone.py:1454-1491`

**BaseModelInference.__init__() - 추론 모델 초기화**
```python
def __init__(
    self,
    ckpt_path,
    configs,
    device,
    save_dir=None,
    unnorm_key: Optional[str] = None,
    policy_setup: str = "widowx_bridge",
    exec_horizon=1,
):
    self.configs = configs
    self.dataset_stat = self.load_dataset_stat()
    self.model = BaseTrainer(configs=configs)
    self.policy = self.model
    
    # 1단계: 환경 변수 설정
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # 2단계: 정책 설정
    if policy_setup == "widowx_bridge":
        unnorm_key = "bridge_orig" if unnorm_key is None else unnorm_key
    elif policy_setup == "google_robot":
        unnorm_key = "fractal20220817_data" if unnorm_key is None else unnorm_key
    
    # 3단계: 그리퍼 액션 설정
    self.sticky_gripper_num_repeat = 2
    self.policy_setup = policy_setup
    self.unnorm_key = unnorm_key
    
    if self.policy_setup == "google_robot":
        self.close_gripper_act = -1
    elif self.policy_setup == "widowx_bridge":
        self.close_gripper_act = 1
    
    # 4단계: 이미지 및 액션 설정
    self.image_size = self.configs.get("image_size", 224)
    self.action_scale = self.configs.get("action_scale", 1.0)
    self.horizon = self.configs["window_size"]
    self.window_size = self.horizon
    self.pred_action_horizon = exec_horizon
```

**출처**: `RoboVLMs/eval/simpler/model_wrapper.py:15-58`

**StandaloneVLAInference.load_model() - 추론 모델 로드**
```python
def load_model(self):
    """VLA 모델 로드"""
    try:
        print(f"📥 모델 로딩 중: {self.model_id}")
        
        model_save_path = Path(self.model_cache_dir) / self.model_id.split('/')[-1]
        model_save_path.mkdir(parents=True, exist_ok=True)

        # 1단계: 프로세서 로드
        self.processor = AutoProcessor.from_pretrained(
            self.model_id, 
            cache_dir=model_save_path
        )

        # 2단계: 모델 로드
        model_kwargs = {
            "cache_dir": model_save_path,
            "low_cpu_mem_usage": True
        }
        
        if self.device.type == "cuda":
            model_kwargs["torch_dtype"] = torch.bfloat16
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["torch_dtype"] = torch.float32

        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            self.model_id, 
            **model_kwargs
        )
        
        if self.device.type != "cuda":
            self.model.to(self.device)
        
        # 3단계: 추론 모드 설정
        self.model.eval()
        print("✅ 모델 로딩 완료")
        
    except Exception as e:
        print(f"❌ 모델 로딩 실패: {e}")
        raise
```

**출처**: `RoboVLMs/vla_test/standalone_vla_test.py:46-85`

### 1.3 학습 vs 추론 변수 비교

| 구분 | 학습 변수 | 추론 변수 |
|------|-----------|-----------|
| **모드** | `model.train()` | `model.eval()` |
| **그래디언트** | `requires_grad=True` | `requires_grad=False` |
| **캐시** | `use_cache=False` | `use_cache=True` |
| **드롭아웃** | 활성화 | 비활성화 |
| **배치 정규화** | 학습 모드 | 평가 모드 |
| **메모리** | 높음 (그래디언트) | 낮음 (그래디언트 없음) |
| **입력** | `action_labels` 포함 | `action_labels` 없음 |
| **출력** | Loss 계산 | 액션 예측만 |
| **토큰 삽입** | Teacher Forcing | Autoregressive |

### 1.4 환경 변수 설정

**Docker 환경 변수 (docker-compose.yml)**
```yaml
environment:
  - DISPLAY=${DISPLAY:-:0}
  - ROS_DOMAIN_ID=42
  - CUDA_VISIBLE_DEVICES=0
  - TORCH_DTYPE=bfloat16
  - PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
  - TRANSFORMERS_CACHE=/workspace/.vlms
  - HF_HOME=/workspace/.vlms
  - PYTHONPATH=/workspace:/workspace/robovlms
  - VLA_MODEL=paligemma-3b-mix-224
  - ACTION_MODE=automotive
  - ACTION_DIM=4
  - WINDOW_SIZE=8
  - INFERENCE_LATENCY_TARGET=100
  - PROJECT_NAME=k_project_event_vla
```

**출처**: `RoboVLMs/docker-compose.yml:25-39`

### 1.5 학습 변수 상세 설정

**CALVIN Fine-tuning 설정**
```json
{
  "train_setup": {
    "precision": "bf16",
    "predict_action": true,
    "predict_forward": false,
    "predict_caption": false,
    "train_vision": true,
    "freeze_backbone": false,
    "freeze_mm_mlp_adapter": false,
    "lora_enable": false,
    "train_text_embedding": true
  },
  "act_head": {
    "type": "LSTMDecoder",
    "hidden_size": 1024,
    "action_dim": 7,
    "down_sample": "none",
    "latent": 1,
    "fwd_pred_next_n": 1,
    "window_size": 1,
    "action_space": "continuous"
  }
}
```

**출처**: `RoboVLMs/README.md:228-267`

## 🤖 2. 실제 로봇 벤치마크 데이터셋 상세 분석

### 2.1 Real-World Experiments 벤치마크

**벤치마크 개요**
- **총 작업 수**: 105개의 조작 작업
- **데이터 규모**: 70,000개 이상의 원격 조작 인간 궤적
- **평가 설정**: 1개 단순 설정 + 4개 도전적 미지 설정
- **총 평가 작업**: 20개 작업
- **롤아웃**: 각 설정당 3회 롤아웃 (작업당 5개 설정)

**출처**: RoboVLMs 논문 Appendix K, Appendix D, Figure 15-17

**로봇 사양**
- **자유도**: 7-DOF (6차원 자세 + 1차원 그리퍼)
- **관측 정보**: 고유 감각 정보 + 시각 관측 + 언어 입력

### 2.2 CALVIN 벤치마크 상세

**CALVIN [32] - Simulated Benchmark**

**데이터셋 구성**
```python
# CALVIN 데이터셋 구조
calvin_dataset = {
    "demonstrations": 24000,                    # 24k 인간 원격 조작 데모
    "trajectory_length": "< 64 timesteps",      # 각 궤적 64 타임스텝 이하
    "language_annotations": True,               # 언어 명령 포함
    "basic_skills": 34,                         # 34개 사전 정의 기본 스킬
    "splits": ["scene_A", "scene_B", "scene_C", "scene_D"]
}
```

**34개 기본 스킬 목록** (15개 태스크 유형 × 색상/방향 조합)

**1-6. Rotate 블록 (6개)**
- Rotate red/blue/pink block right: z축 기준 시계방향 60도 이상 회전 (x/y축 30도 이내)
- Rotate red/blue/pink block left: z축 기준 반시계방향 60도 이상 회전 (x/y축 30도 이내)

**7-12. Push 블록 (6개)**
- Push red/blue/pink block right: 블록을 오른쪽으로 10cm 이상 이동 (양쪽 프레임에서 표면 접촉 유지)
- Push red/blue/pink block left: 블록을 왼쪽으로 10cm 이상 이동 (양쪽 프레임에서 표면 접촉 유지)

**13-14. Move slider (2개)**
- Move slider left/right: 슬라이딩 도어를 최소 12cm 밀기

**15-16. Drawer 조작 (2개)**
- Open/close drawer: 서랍을 최소 10cm 밀어넣기/당기기

**17-19. Lift block table (3개)**
- Lift red/blue/pink block table: 테이블 표면에서 블록을 잡아 최소 5cm 들어올리기
  (첫 프레임에서 그리퍼는 물체를 터치하지 않음)

**20-22. Lift block slider (3개)**
- Lift red/blue/pink block slider: 슬라이딩 캐비닛 표면에서 블록을 잡아 최소 3cm 들어올리기
  (첫 프레임에서 그리퍼는 물체를 터치하지 않음)

**23-25. Lift block drawer (3개)**
- Lift red/blue/pink block drawer: 서랍 표면에서 블록을 잡아 최소 5cm 들어올리기
  (첫 프레임에서 그리퍼는 물체를 터치하지 않음)

**26. Place in slider/drawer (1개)**
- Place in slider/drawer: 슬라이딩 캐비닛/서랍에 물체를 넣기
  (첫 프레임에서 그리퍼가 물체를 들고 있어야 함)

**27. Push into drawer (1개)**
- Push into drawer: 서랍에 물체를 밀어넣기
  (첫 프레임에서 테이블 표면의 물체를 터치해야 함)

**28. Stack blocks (1개)**
- Stack blocks: 한 블록을 다른 블록 위에 쌓기
  (최종 프레임에서 그리퍼가 블록과 접촉하지 않음)

**29. Unstack blocks (1개)**
- Unstack blocks: 다른 블록 위에서 블록을 제거
  (최종 프레임에서 그리퍼가 블록과 접촉하지 않음)

**30-31. Light bulb (2개)**
- Turn on/off light bulb: 노란색 전구를 켜기/끄기 위해 스위치를 위/아래로 누르기

**32-33. LED (2개)**
- Turn on/off LED: 초록색 LED 라이트를 켜기/끄기 위해 버튼을 누르기

**총 34개 스킬** = Rotate(6) + Push(6) + Slider(2) + Drawer(2) + Lift table(3) + Lift slider(3) + Lift drawer(3) + Place(1) + Push into(1) + Stack(1) + Unstack(1) + Light(2) + LED(2) + Open oven(1) = 34개

**평가 메트릭**
- **Sequential Task Success Rate**: 5개 연속 작업 완료 성공률
- **Average Length**: 달성한 작업의 평균 길이
- **평가 규모**: D split에서 1000 롤아웃, 각 롤아웃당 5개 연속 서브태스크

**출처**: CALVIN 논문 [32], RoboVLMs 논문

### 2.3 SimplerEnv 벤치마크

**SimplerEnv [25] - Real-to-Sim Evaluation**

**벤치마크 목적**
- 실제 로봇 정책을 시뮬레이션에서 평가
- Google Robot, BridgeData V2와 비교 가능한 아레나 제공
- 효율적이고 확장 가능한 실제 세계 평가 대안

#### 2.3.1 Google Robot 설정 작업

**1) pick coke can**
```python
# pick coke can 작업 설정
task_config = {
    "objective": "빈 코크 캔을 테이블에서 집어 들기",
    "positions": ["horizontal", "vertical", "upright"],  # 3가지 위치
    "grid_points": 25,                                   # 직사각형 영역 내 25개 그리드
    "total_trials": 75,                                  # 25 × 3 = 75 시험
    "distractors": False                                 # 표준 설정에서는 방해 요소 없음
}
```

**2) move {obj1} near {obj2}**
```python
# move near 작업 설정
task_config = {
    "objective": "obj1을 obj2 근처로 이동",
    "objects": ["blue plastic bottle", "Pepsi can", "orange", 
                "7up can", "apple", "sponge", "Coke can", "Redbull can"],  # 8개 물체
    "formation": "triangular",                           # 삼각형 배치
    "triplets": 5,                                       # 5개 triplet (랜덤 선택)
    "patterns": ["upright", "inverted"],                 # 2가지 삼각형 패턴
    "trials_per_triplet": 6,                            # triplet당 6회 시험
    "total_trials": 60                                   # 5 × 6 × 2 = 60 시험
}
```

**3) (open/close) (top/middle/bottom) drawer**
```python
# drawer 작업 설정
task_config = {
    "objective": "특정 서랍 열기/닫기",
    "drawers": 3,                                        # top, middle, bottom
    "actions": ["open", "close"],                        # 2가지 액션
    "robot_positions": 9,                                # 9개 그리드 위치
    "total_trials": 54,                                  # 3 × 2 × 9 = 54 시험
    "evaluation_type": "articulated_objects"             # 관절 물체 처리 능력 평가
}
```

**4) open top drawer; place apple into top drawer**
```python
# multi-step 작업 설정
task_config = {
    "objective": "서랍 열고 사과를 서랍에 넣기",
    "steps": [
        "open top drawer",
        "place apple into top drawer"
    ],
    "robot_positions": 3,                                # 로봇 위치 3개
    "apple_positions": 9,                                # 사과 그리드 위치 9개
    "total_trials": 27,                                  # 3 × 9 = 27 시험
    "instruction_switch": "midpoint or terminate token", # 명령 전환 시점
    "evaluation_type": "sequential_multi-action"         # 순차적 다중 액션 평가
}
```

#### 2.3.2 WidowX + Bridge 설정 작업

**1) put the spoon on the towel**
```python
# spoon on towel 작업 설정
task_config = {
    "objective": "수저를 타월 위에 놓기",
    "square_size": "15 cm",                              # 정사각형 크기
    "spoon_positions": ["corner_1", "corner_2", "corner_3", "corner_4"],
    "towel_positions": ["corner_1", "corner_2", "corner_3", "corner_4"],
    "spoon_orientations": ["horizontal", "vertical"],    # 2가지 방향
    "total_trials": 24,                                  # 4 × 4 × 2 / 2 = 24 시험
    "gripper_adjustment": True                           # 그리퍼 방향 조정 필요
}
```

**2) put carrot on plate**
```python
# carrot on plate 작업 설정
task_config = {
    "objective": "당근을 접시 위에 놓기",
    "square_size": "15 cm",
    "carrot_positions": ["corner_1", "corner_2", "corner_3", "corner_4"],
    "plate_positions": ["corner_1", "corner_2", "corner_3", "corner_4"],
    "total_trials": 24,
    "similar_to": "put the spoon on the towel"
}
```

**3) stack the green block on the yellow block**
```python
# block stacking 작업 설정
task_config = {
    "objective": "초록 블록을 노란 블록 위에 쌓기",
    "block_size": "3 cm",                                # 블록 크기
    "square_configs": [
        {"size": "10 cm", "trials": 12},                 # 10cm 정사각형
        {"size": "20 cm", "trials": 12}                  # 20cm 정사각형
    ],
    "green_block_positions": 4,                          # 4개 코너
    "yellow_block_positions": 4,                         # 4개 코너
    "total_trials": 24                                   # (4 × 4 / 2) × 2 = 24 시험
}
```

**4) put eggplant into yellow basket**
```python
# eggplant into basket 작업 설정
task_config = {
    "objective": "가지를 노란 바구니에 넣기",
    "environment": "sink with two basins",               # 2개 세면대
    "eggplant_location": "right basin (random)",         # 오른쪽 세면대 (랜덤 위치)
    "basket_location": "left basin",                     # 왼쪽 세면대
    "eggplant_variations": {
        "position": "random",
        "orientation": "random",
        "constraint": "easily graspable, away from edges"
    },
    "total_trials": 24
}
```

### 2.4 벤치마크 평가 요약

| 벤치마크 | 유형 | 작업 수 | 데이터 규모 | 평가 메트릭 |
|---------|------|---------|-------------|-------------|
| **Real-World Experiments** | 실제 로봇 | 20개 (105개 중) | 70,000+ 궤적 | 설정별 평균 성공률 |
| **CALVIN** | 시뮬레이션 | 34개 기본 스킬 | 24,000 데모 | Sequential Success Rate, Avg Length |
| **SimplerEnv (Google)** | Real-to-Sim | 4개 작업 | - | 시험별 성공률 (75-54회) |
| **SimplerEnv (Bridge)** | Real-to-Sim | 4개 작업 | - | 시험별 성공률 (24회) |

### 2.5 코드 구현 예시

**DiskCalvinDataset - CALVIN 데이터 로딩**
```python
class DiskCalvinDataset(BaseCalvinDataset):
    """디스크에서 개별 파일로 에피소드를 로드하는 데이터셋"""
    def __init__(
        self,
        image_fn: Callable,
        tokenizer: Callable,
        *args: Any,
        skip_frames: int = 1,
        seq_len: int = 1,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        # ... (초기화 코드)
```

**출처**: `RoboVLMs/robovlms/data/calvin_dataset.py:428-447`

**SimplerEnv 평가 함수**
```python
def evaluate_simpler_env(model, env, task_config):
    """SimplerEnv에서 모델 평가"""
    success_count = 0
    total_trials = task_config["total_trials"]
    
    for trial in range(total_trials):
        # 환경 초기화
        obs = env.reset()
        
        # 모델 추론
        action = model.inference(
            vision_x=obs["rgb"],
            lang_x=task_config["instruction"]
        )
        
        # 액션 실행 및 평가
        success = env.step(action)
        success_count += int(success)
    
    success_rate = success_count / total_trials
    return success_rate
```

## 🎯 3. 핵심 학습 아이디어

### 3.1 VLM의 역할

**1) 멀티모달 이해**
- 이미지와 텍스트를 동시에 이해
- 로봇 환경의 시각적 상황 파악
- 언어 명령의 의미 해석

**2) 특징 추출**
- Vision Encoder: 이미지에서 시각적 특징 추출
- Language Encoder: 텍스트에서 언어적 특징 추출
- Cross-modal Fusion: 비전-언어 특징 융합

**3) Fine-tuning 목적**
- 로봇 도메인에 특화된 표현 학습
- 액션과 관련된 시각적/언어적 특징 강화
- Policy Head를 위한 고품질 특징 제공

### 3.2 LSTM Layer의 역할

**1) 시퀀스 처리**
- 시간적 연속성 모델링
- 히스토리 정보 활용
- 동적 상태 추적

**2) 액션 예측**
- VLM 특징을 액션 공간으로 매핑
- 6-DOF 팔 액션 + 1-DOF 그리퍼 예측
- 연속적이고 부드러운 액션 생성

**3) 학습 목적**
- VLM 특징과 액션 간의 관계 학습
- 최적의 액션 정책 학습
- 로봇 제어에 특화된 표현 학습

### 3.3 동시 학습 메커니즘

**End-to-End 학습**
```python
# VLM + LSTM 동시 학습
loss_total = loss_vlm + loss_action

# VLM Loss: 멀티모달 표현 학습
loss_vlm = calculate_vl_cross_entropy(vlm_logits, text_labels)

# Action Loss: 액션 예측 학습
loss_action = loss_arm + 0.01 * loss_gripper
```

**장점**
1. **통합 최적화**: VLM과 LSTM이 함께 최적화
2. **특징 품질**: 액션 예측에 유용한 특징 학습
3. **효율성**: 별도 학습 대비 시간/자원 절약

### 3.4 학습 vs 추론 차이

**학습 시 (Training)**
- Action Labels 제공
- Loss 계산 및 Backpropagation
- Gradient 업데이트
- Teacher Forcing (Discrete Action)

**추론 시 (Inference)**
- Action Labels 없음
- Action 예측만 수행
- Gradient 계산 없음
- Autoregressive Generation (Discrete Action)

## 📊 4. 전체 시스템 요약

### 4.1 학습 파이프라인

```
[Real-World Data]
    ↓
[CALVIN/Bridge Dataset]
    ↓
[Data Preprocessing]
    ↓
[VLM Fine-tuning] ← Full-FT or LoRA
    ↓
[LSTM Training] ← Action Prediction
    ↓
[Loss Calculation] ← VL Loss + Action Loss
    ↓
[Optimization] ← Adam/AdamW
    ↓
[Trained Model]
```

### 4.2 추론 파이프라인

```
[Robot Camera] → [Image] → [VLM Encoder]
                                ↓
[Language Command] → [Text] → [VLM Encoder]
                                ↓
                        [Multimodal Fusion]
                                ↓
                        [LSTM Decoder]
                                ↓
                        [Action Prediction]
                                ↓
                        [Robot Control]
```

### 4.3 핵심 성능 지표

| 벤치마크 | 평가 메트릭 | 목표 성능 |
|---------|-------------|-----------|
| **CALVIN** | Sequential Success Rate | 5개 연속 작업 완료율 |
| **SimplerEnv** | Task Success Rate | 개별 작업 성공률 |
| **Real-World** | Rollout Success Rate | 실제 환경 성공률 |

### 4.4 RoboVLMs 특징

**1) 다양한 VLM 백본 지원**
- PaliGemma, Flamingo, Kosmos, Qwen-VL 등

**2) 유연한 Policy Head**
- LSTMDecoder, FCDecoder, GPTDecoder, DiscreteDecoder

**3) 효율적인 학습**
- Full Fine-tuning과 LoRA 모두 지원
- BFloat16 정밀도로 메모리 효율성
- Gradient Checkpointing으로 대규모 모델 학습

**4) 실용적인 평가**
- CALVIN, SimplerEnv, Real-World 벤치마크
- 다양한 난이도와 설정에서 평가
- 체계적인 성능 측정

## 📚 참고 자료

**출처 논문**
- RoboVLMs: "Towards Generalist Robot Policies: What Matters in Building Vision-Language-Action Models"
- CALVIN [32]: "CALVIN: A Benchmark for Language-Conditioned Policy Learning for Long-Horizon Robot Manipulation Tasks"
- SimplerEnv [25]: Real-to-Sim Evaluation Framework

**GitHub 저장소**
- RoboVLMs: https://github.com/RoboVLMs/RoboVLMs
- CALVIN Dataset: https://github.com/mees/calvin

