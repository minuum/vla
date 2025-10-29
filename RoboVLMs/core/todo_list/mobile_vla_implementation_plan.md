# Mobile VLA 구현 계획 및 Todo List

## 1. 학습 방법 결정 (LoRA vs Full Fine-tuning)

### 1.1 논문 분석 결과

**기존 RoboVLMs 논문**:
- **Full Fine-tuning 사용**: VLM 전체 파라미터 학습
- **메모리 요구사항**: 24GB+ GPU
- **성능**: 최고 성능 달성

**우리 Mobile VLA 요구사항**:
- **Jetson AGX Orin**: 16GB 메모리 제한
- **실시간 추론**: Mobile robot에 적합한 속도
- **전력 효율성**: 60W 전력 제한

### 1.2 LoRA vs Full Fine-tuning 비교

| **구분** | **Full Fine-tuning** | **LoRA Fine-tuning** |
|----------|---------------------|---------------------|
| **메모리 사용량** | 24GB+ | 8-16GB |
| **학습 속도** | 느림 | 빠름 |
| **성능** | 최고 | 90-95% |
| **Jetson 호환성** | 불가능 | 가능 |
| **전력 소비** | 높음 | 낮음 |

### 1.3 결론: LoRA Fine-tuning 선택

**이유**:
1. **Jetson 호환성**: 16GB 메모리 제한 내에서 동작
2. **전력 효율성**: 60W 전력 제한 내에서 동작
3. **학습 속도**: 빠른 학습으로 빠른 프로토타이핑
4. **성능**: 90-95% 성능으로 충분

## 2. 1000개 Dataset 수집 계획

### 2.1 데이터 수집 전략

**데이터 구성**:
- **총 에피소드**: 1,000개
- **에피소드당 길이**: 50-100 스텝
- **총 스텝 수**: 50,000-100,000 스텝
- **수집 기간**: 2-3주

**태스크 분류**:
```python
TASK_CATEGORIES = {
    "navigation": {
        "count": 400,           # 40%
        "tasks": [
            "go to kitchen",
            "go to living room",
            "go to bedroom",
            "navigate to object"
        ]
    },
    "manipulation": {
        "count": 300,           # 30%
        "tasks": [
            "pick up object",
            "place object",
            "open drawer",
            "close drawer"
        ]
    },
    "combined": {
        "count": 300,           # 30%
        "tasks": [
            "go to kitchen and pick up cup",
            "navigate to table and place object",
            "go to bedroom and open drawer"
        ]
    }
}
```

### 2.2 데이터 수집 환경

**하드웨어 설정**:
```python
HARDWARE_CONFIG = {
    "robot": "Mobile Robot (2D movement)",
    "cameras": {
        "static_camera": "RGB camera (224x224)",
        "gripper_camera": "RGB camera (224x224, optional)"
    },
    "sensors": {
        "odometry": "2D position (x, y)",
        "gripper_state": "open/close"
    }
}
```

**환경 설정**:
```python
ENVIRONMENT_CONFIG = {
    "rooms": ["kitchen", "living_room", "bedroom", "bathroom"],
    "objects": ["cup", "book", "phone", "remote", "bottle"],
    "furniture": ["table", "chair", "sofa", "bed", "desk"],
    "lighting": ["day", "night", "artificial"]
}
```

### 2.3 데이터 수집 프로세스

**1단계: 환경 준비**
```python
def setup_data_collection_environment():
    # 1. 로봇 하드웨어 설정
    robot = MobileRobot()
    robot.initialize()
    
    # 2. 카메라 설정
    static_camera = Camera(position="ceiling")
    gripper_camera = Camera(position="robot_gripper")
    
    # 3. 데이터 저장 설정
    data_storage = HDF5Storage("mobile_vla_dataset.h5")
    
    return robot, static_camera, gripper_camera, data_storage
```

**2단계: 에피소드 수집**
```python
def collect_episode(episode_id, task_description):
    episode_data = {
        "episode_id": episode_id,
        "task": task_description,
        "images": [],
        "actions": [],
        "states": [],
        "language": task_description
    }
    
    # 에피소드 시작
    for timestep in range(episode_length):
        # 1. 이미지 캡처
        static_img = static_camera.capture()
        gripper_img = gripper_camera.capture()
        
        # 2. 로봇 상태 읽기
        robot_state = robot.get_state()  # [x, y, gripper_state]
        
        # 3. 액션 실행 (전문가 조작)
        action = expert_control()  # [delta_x, delta_y, gripper_action]
        
        # 4. 데이터 저장
        episode_data["images"].append([static_img, gripper_img])
        episode_data["actions"].append(action)
        episode_data["states"].append(robot_state)
    
    return episode_data
```

**3단계: 데이터 검증**
```python
def validate_episode(episode_data):
    """에피소드 데이터 검증"""
    # 이미지 검증
    assert len(episode_data["images"]) > 0, "No images found"
    assert all(img.shape == (2, 224, 224, 3) for img in episode_data["images"]), "Invalid image shape"
    
    # 액션 검증
    assert len(episode_data["actions"]) > 0, "No actions found"
    assert all(action.shape == (3,) for action in episode_data["actions"]), "Invalid action shape"
    
    # 상태 검증
    assert len(episode_data["states"]) > 0, "No states found"
    assert all(state.shape == (3,) for state in episode_data["states"]), "Invalid state shape"
    
    # 언어 명령 검증
    assert episode_data["language"] != "", "No language command found"
    
    return True
```

## 3. LSTM Layer 학습 계획

### 3.1 LSTM 구조 설계

**Mobile VLA용 LSTM 설정**:
```python
LSTM_CONFIG = {
    "input_features": 1024,        # VLM 출력 차원
    "action_dim": 3,               # 2D 액션 (X, Y, Gripper)
    "hidden_size": 512,            # Jetson에 적합한 크기
    "num_layers": 2,               # 2층 LSTM (메모리 절약)
    "window_size": 8,              # 히스토리 길이
    "fwd_pred_next_n": 5,          # 예측할 액션 수 (Mobile robot에 적합)
    "dropout": 0.1,                # 과적합 방지
    "bidirectional": False         # 단방향 LSTM
}
```

**LSTM 구현**:
```python
class MobileVLALSTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # LSTM 네트워크
        self.lstm = nn.LSTM(
            input_size=config["input_features"],
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
            batch_first=True
        )
        
        # Action Head (2D + Gripper)
        self.action_head = nn.Linear(
            config["hidden_size"],
            config["fwd_pred_next_n"] * config["action_dim"]
        )
        
    def forward(self, x):
        # LSTM Forward
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Action 예측
        actions = self.action_head(lstm_out)
        
        # Reshape: [batch, window, fwd_pred_next_n, action_dim]
        actions = actions.view(
            -1, self.config["window_size"], 
            self.config["fwd_pred_next_n"], 
            self.config["action_dim"]
        )
        
        return actions
```

### 3.2 Loss 함수 설계

**Mobile VLA용 Loss 함수**:
```python
def mobile_vla_loss(pred_actions, target_actions, attention_mask=None):
    """
    pred_actions: [batch, window, fwd_pred_next_n, 3]
    target_actions: [batch, window, fwd_pred_next_n, 3]
    attention_mask: [batch, window, fwd_pred_next_n]
    """
    # 2D Movement Loss (MSE)
    movement_loss = F.mse_loss(
        pred_actions[..., :2],  # X, Y
        target_actions[..., :2]
    )
    
    # Gripper Loss (BCE)
    gripper_loss = F.binary_cross_entropy_with_logits(
        pred_actions[..., 2],   # Gripper
        target_actions[..., 2]
    )
    
    # 가중치 조합
    total_loss = movement_loss + 0.1 * gripper_loss
    
    return {
        "total_loss": total_loss,
        "movement_loss": movement_loss,
        "gripper_loss": gripper_loss
    }
```

### 3.3 학습 설정

**LoRA + LSTM 학습 설정**:
```python
TRAINING_CONFIG = {
    "model": {
        "vlm_backbone": "kosmos-2",
        "freeze_backbone": True,
        "lora_enable": True,
        "lora_r": 32,              # Jetson에 적합한 크기
        "lora_alpha": 16,
        "lora_dropout": 0.1
    },
    "lstm": {
        "hidden_size": 512,
        "num_layers": 2,
        "dropout": 0.1
    },
    "training": {
        "learning_rate": 1e-4,
        "batch_size": 2,           # Jetson 메모리 제한
        "max_epochs": 10,
        "gradient_clip_val": 1.0,
        "precision": "fp16"        # 메모리 절약
    }
}
```

## 4. 추론 코드 및 컨테이너 문제 해결

### 4.1 Jetson 최적화 추론 코드

**실시간 추론 구현**:
```python
class MobileVLAInference:
    def __init__(self, model_path, device="cuda"):
        self.device = device
        self.model = self.load_model(model_path)
        self.history_memory = []
        
    def load_model(self, model_path):
        """모델 로드 및 최적화"""
        # 모델 로드
        model = MobileVLAModel.from_pretrained(model_path)
        model.eval()
        
        # Jetson 최적화
        model = model.to(self.device)
        model = torch.jit.script(model)  # TorchScript 최적화
        
        return model
    
    def predict_action(self, image, text):
        """실시간 액션 예측"""
        with torch.no_grad():
            # 1. 이미지 전처리
            processed_image = self.preprocess_image(image)
            
            # 2. 텍스트 토큰화
            tokenized_text = self.tokenize_text(text)
            
            # 3. 히스토리 관리
            self.update_history(processed_image, tokenized_text)
            
            # 4. 액션 예측
            if len(self.history_memory) >= self.window_size:
                action = self.model(
                    self.history_memory[-self.window_size:],
                    tokenized_text
                )
                return action[0]  # 첫 번째 액션만 반환
            else:
                return None  # 히스토리 부족
```

### 4.2 Docker 컨테이너 최적화

**Jetson용 Dockerfile**:
```dockerfile
# Jetson L4T 기반 이미지
FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

# 시스템 패키지 업데이트
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    libjpeg-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Python 패키지 설치
RUN pip3 install --no-cache-dir \
    torch==2.0.0 \
    torchvision==0.15.0 \
    torchaudio==2.0.0 \
    transformers==4.30.0 \
    peft==0.4.0 \
    einops==0.6.1 \
    pillow==9.5.0

# RoboVLMs 설치
COPY . /workspace/robovlms
WORKDIR /workspace/robovlms
RUN pip3 install -e .

# 메모리 최적화 설정
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
ENV CUDA_VISIBLE_DEVICES=0

# 실행 스크립트
COPY mobile_vla_inference.py /workspace/
CMD ["python3", "/workspace/mobile_vla_inference.py"]
```

**Docker Compose 설정**:
```yaml
version: '3.8'
services:
  mobile-vla:
    build: .
    container_name: mobile-vla-inference
    runtime: nvidia
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
    volumes:
      - ./models:/workspace/models
      - ./data:/workspace/data
    ports:
      - "8080:8080"
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 14G  # Jetson 16GB 중 14GB 사용
```

### 4.3 메모리 최적화

**Jetson 메모리 최적화**:
```python
def optimize_memory_usage():
    """Jetson 메모리 최적화"""
    # 1. CUDA 메모리 할당 최적화
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(0.9)  # 90% 사용
    
    # 2. 모델 최적화
    model = torch.jit.script(model)  # TorchScript 최적화
    model = torch.jit.optimize_for_inference(model)  # 추론 최적화
    
    # 3. 배치 크기 조정
    batch_size = 1  # 실시간 추론용
    
    # 4. Mixed Precision 사용
    model = model.half()  # FP16 사용
    
    return model
```

## 5. 구현 일정

### 5.1 1주차: 데이터 수집 환경 구축

**목표**: 데이터 수집 환경 구축 및 첫 100개 에피소드 수집

**작업 내용**:
- [ ] 로봇 하드웨어 설정
- [ ] 카메라 시스템 구축
- [ ] 데이터 수집 스크립트 개발
- [ ] 첫 100개 에피소드 수집
- [ ] 데이터 검증 및 전처리

### 5.2 2주차: 데이터 수집 완료

**목표**: 1000개 에피소드 수집 완료

**작업 내용**:
- [ ] 나머지 900개 에피소드 수집
- [ ] 데이터 품질 검증
- [ ] 데이터셋 전처리 및 정규화
- [ ] 학습/검증/테스트 분할

### 5.3 3주차: LoRA Fine-tuning 구현

**목표**: LoRA Fine-tuning 코드 구현 및 테스트

**작업 내용**:
- [ ] LoRA 설정 구현
- [ ] Mobile VLA 모델 구조 설계
- [ ] 학습 스크립트 개발
- [ ] Jetson에서 학습 테스트

### 5.4 4주차: LSTM Layer 학습

**목표**: LSTM Layer 학습 및 최적화

**작업 내용**:
- [ ] LSTM 구조 구현
- [ ] Loss 함수 설계
- [ ] 학습 파이프라인 구축
- [ ] 하이퍼파라미터 튜닝

### 5.5 5주차: 추론 코드 및 컨테이너 최적화

**목표**: 실시간 추론 시스템 구축

**작업 내용**:
- [ ] 실시간 추론 코드 개발
- [ ] Jetson 최적화
- [ ] Docker 컨테이너 구축
- [ ] 성능 테스트 및 최적화

## 6. 성공 지표

### 6.1 학습 성공 지표

**데이터 수집**:
- [ ] 1000개 에피소드 수집 완료
- [ ] 데이터 품질 검증 통과
- [ ] 학습/검증/테스트 분할 완료

**모델 학습**:
- [ ] LoRA Fine-tuning 수렴
- [ ] LSTM Loss 감소 확인
- [ ] 검증 정확도 80% 이상

### 6.2 추론 성공 지표

**성능 지표**:
- [ ] 추론 속도: 10 FPS 이상
- [ ] 메모리 사용량: 14GB 이하
- [ ] 정확도: 75% 이상

**시스템 지표**:
- [ ] Docker 컨테이너 정상 동작
- [ ] Jetson에서 안정적 실행
- [ ] 실시간 추론 가능

## 7. 위험 요소 및 대응 방안

### 7.1 위험 요소

**데이터 수집**:
- **위험**: 데이터 품질 부족
- **대응**: 엄격한 검증 기준 설정

**학습**:
- **위험**: Jetson 메모리 부족
- **대응**: 모델 크기 최적화, Mixed Precision 사용

**추론**:
- **위험**: 실시간 성능 부족
- **대응**: TorchScript 최적화, 배치 크기 조정

### 7.2 대응 방안

**메모리 부족 시**:
1. 모델 크기 축소 (LoRA rank 감소)
2. Mixed Precision 사용
3. Gradient Checkpointing 활성화

**성능 부족 시**:
1. TorchScript 최적화
2. 배치 크기 조정
3. 모델 양자화 고려

**정확도 부족 시**:
1. 데이터 증강 강화
2. 하이퍼파라미터 튜닝
3. 모델 구조 개선

## 8. 참고 자료

- `RoboVLMs/robovlms/model/backbone/base_backbone.py`: VLM + LSTM 통합
- `RoboVLMs/robovlms/model/policy_head/base_policy.py`: LSTM 구현
- `RoboVLMs/robovlms/train/base_trainer.py`: 학습 로직
- `RoboVLMs/configs/calvin_finetune/`: CALVIN 설정
- `RoboVLMs/configs/mobile_vla/`: Mobile VLA 설정
