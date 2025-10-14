# RoboVLMs Implementation Analysis

## 설치 및 설정

### 1. 환경 설정

#### CALVIN 시뮬레이션
```bash
# Python 3.8.10 환경 생성
conda create -n robovlms python=3.8.10 -y

# SIMPLER 시뮬레이션
conda create -n robovlms python=3.10 -y

# 환경 활성화
conda activate robovlms

# CUDA 툴킷 설치
conda install cudatoolkit cudatoolkit-dev -y

# RoboVLMs 설치
pip install -e .
```

#### OXE 데이터셋 훈련
```bash
# OpenVLA 포크 클론
git clone https://github.com/lixinghang12/openvla
cd openvla
pip install -e .
```

### 2. 벤치마크 환경 설정

#### CALVIN 설치
```bash
# CALVIN 설치 스크립트
bash scripts/setup_calvin.sh

# 설치 검증
python eval/calvin/env_test.py
```

#### SimplerEnv 설치
```bash
# SimplerEnv 설치 스크립트
bash scripts/setup_simplerenv.sh

# 설치 검증
python eval/simpler/env_test.py
```

## VLM 통합 튜토리얼

### 1. VLM 속성 설정

#### 필수 속성 구성
```python
class RoboCustomVLM(BaseRoboVLM):
    @property
    def image_processor(self):
        """이미지 전처리기 반환"""
        return self.model.processor
    
    @property
    def hidden_size(self):
        """VLM 백본의 히든 크기 반환"""
        return self.model.config.hidden_size
    
    @property
    def word_embedding(self):
        """단어 임베딩 반환"""
        return self.model.embed_tokens
    
    @property
    def text_tower(self):
        """텍스트 처리 컴포넌트 반환"""
        return self.model.language_model
    
    @property
    def vision_tower(self):
        """비전 처리 컴포넌트 반환"""
        return self.model.vision_tower
    
    @property
    def model(self):
        """VLM 백본 반환"""
        return self.backbone
    
    def model_encode_images(self, images):
        """이미지를 비전 토큰으로 인코딩"""
        # 비전 인코더로 이미지 처리
        image_features = self.vision_tower(images)
        
        # 멀티모달 프로젝터로 특징 변환
        image_tokens = self.multi_modal_projector(image_features)
        
        # 정규화
        image_tokens = image_tokens / (self.hidden_size ** 0.5)
        
        return image_tokens
```

### 2. VLA 등록

#### 백본 등록
```python
# model/backbone/__init__.py 파일 수정
from .robocustomvlm import RoboCustomVLM
__all__.append('RoboCustomVLM')
```

#### 설정 파일 업데이트
```python
# configs/custom_vlm_config.yaml
model:
  backbone: "RoboCustomVLM"
  hidden_size: 4096
  action_dim: 7
  history_length: 16
  action_chunk_size: 10
```

## 훈련 파이프라인

### 1. 데이터 준비

#### CALVIN 데이터셋
```python
# CALVIN 데이터셋 로드
from robovlms.data.calvin import CalvinDataset

dataset = CalvinDataset(
    data_path="/path/to/calvin/data",
    split="ABCD",  # 또는 "ABC"
    window_size=16,
    action_chunk_size=10
)

dataloader = DataLoader(
    dataset, 
    batch_size=128, 
    shuffle=True,
    num_workers=4
)
```

#### SimplerEnv 데이터셋
```python
# SimplerEnv 데이터셋 로드
from robovlms.data.simpler import SimplerDataset

dataset = SimplerDataset(
    data_path="/path/to/simpler/data",
    environment="widowx_bridge",  # 또는 "google_robot"
    window_size=16,
    action_chunk_size=10
)
```

### 2. 모델 훈련

#### 기본 훈련 스크립트
```python
# train.py
import torch
from robovlms.models import RoboVLMs
from robovlms.trainer import VLATrainer

# 모델 초기화
model = RoboVLMs(
    backbone="KosMos",
    architecture="policy_head",
    action_space="continuous"
)

# 훈련기 초기화
trainer = VLATrainer(
    model=model,
    learning_rate=1e-4,
    weight_decay=0.01,
    warmup_ratio=0.25
)

# 훈련 실행
trainer.train(
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    epochs=5,
    save_path="./checkpoints"
)
```

#### 하이퍼파라미터 튜닝
```python
# 하이퍼파라미터 그리드 서치
hyperparameter_grid = {
    'learning_rate': [1e-4, 2e-5, 1e-5],
    'weight_decay': [0, 1e-1],
    'batch_size': [128, 256, 512]
}

best_config = trainer.grid_search(
    hyperparameter_grid,
    train_dataloader,
    val_dataloader
)
```

### 3. Cross-embodiment 훈련

#### OXE 데이터셋 훈련
```python
# OXE 데이터셋으로 사전 훈련
from robovlms.data.oxe import OXEDataset

oxe_dataset = OXEDataset(
    data_path="/path/to/oxe/data",
    window_size=16,
    action_chunk_size=10
)

# 사전 훈련
trainer.pretrain(
    dataloader=oxe_dataloader,
    epochs=10,
    save_path="./pretrained_models"
)

# 도메인 내 파인튜닝
trainer.finetune(
    dataloader=domain_dataloader,
    epochs=5,
    pretrained_path="./pretrained_models/best_model.pt"
)
```

## 평가 파이프라인

### 1. 시뮬레이션 평가

#### CALVIN 평가
```python
# CALVIN 평가 스크립트
from robovlms.eval.calvin import CalvinEvaluator

evaluator = CalvinEvaluator(
    model=model,
    env_path="/path/to/calvin/env",
    split="D"
)

# 평가 실행
results = evaluator.evaluate(
    num_rollouts=1000,
    consecutive_tasks=5
)

print(f"Average Length: {results['avg_length']}")
print(f"Success Rates: {results['success_rates']}")
```

#### SimplerEnv 평가
```python
# SimplerEnv 평가 스크립트
from robovlms.eval.simpler import SimplerEvaluator

evaluator = SimplerEvaluator(
    model=model,
    env_path="/path/to/simpler/env",
    environment="widowx_bridge"
)

# 평가 실행
results = evaluator.evaluate(
    tasks=["put_spoon_on_towel", "put_carrot_on_plate"],
    num_rollouts=24
)

print(f"Task Success Rates: {results['task_success_rates']}")
```

### 2. 실제 로봇 평가

#### 로봇 설정
```python
# 실제 로봇 평가 설정
from robovlms.eval.real_robot import RealRobotEvaluator

evaluator = RealRobotEvaluator(
    model=model,
    robot_config={
        'robot_type': 'kinova_gen3',
        'gripper_type': 'robotiq_2f85',
        'cameras': ['kinect_azure', 'realsense_d435i']
    }
)

# 평가 실행
results = evaluator.evaluate(
    tasks=["open_drawer", "pickup_eggplant", "press_toaster"],
    settings=["simple", "unseen_distractor", "unseen_background"],
    num_rollouts=5
)
```

## 모델 배포

### 1. 모델 저장 및 로드

#### 모델 저장
```python
# 모델 저장
torch.save({
    'model_state_dict': model.state_dict(),
    'config': model.config,
    'performance': evaluation_results
}, 'best_model.pt')
```

#### 모델 로드
```python
# 모델 로드
checkpoint = torch.load('best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

### 2. 추론 파이프라인

#### 단일 추론
```python
# 단일 추론
def predict_action(model, image, language_instruction):
    with torch.no_grad():
        # 이미지 전처리
        processed_image = model.image_processor(image)
        
        # 액션 예측
        action = model.predict_action(
            vision_x=processed_image,
            lang_x=language_instruction
        )
        
        return action
```

#### 배치 추론
```python
# 배치 추론
def predict_actions_batch(model, images, language_instructions):
    with torch.no_grad():
        # 배치 처리
        processed_images = model.image_processor(images)
        
        # 액션 예측
        actions = model.predict_actions(
            vision_x=processed_images,
            lang_x=language_instructions
        )
        
        return actions
```

### 3. 실시간 제어

#### ROS 통합
```python
# ROS 노드로 실시간 제어
import rospy
from robovlms.ros import RoboVLMsNode

class RoboVLMsController:
    def __init__(self, model_path):
        self.model = self.load_model(model_path)
        self.robot = self.initialize_robot()
    
    def control_loop(self):
        while not rospy.is_shutdown():
            # 이미지 및 언어 지시 수신
            image = self.get_current_image()
            instruction = self.get_current_instruction()
            
            # 액션 예측
            action = self.model.predict_action(image, instruction)
            
            # 로봇에 액션 전송
            self.robot.execute_action(action)
            
            rospy.sleep(0.1)  # 10Hz 제어 주기
```

## 성능 최적화

### 1. 메모리 최적화

#### 그래디언트 체크포인팅
```python
# 그래디언트 체크포인팅으로 메모리 사용량 감소
from torch.utils.checkpoint import checkpoint

def memory_efficient_forward(self, x):
    return checkpoint(self._forward, x)

def _forward(self, x):
    # 실제 forward pass
    return self.model(x)
```

#### 모델 병렬화
```python
# 모델을 여러 GPU에 분산
model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])

# 배치를 여러 GPU에 분산
batch_size_per_gpu = batch_size // num_gpus
for gpu_id in range(num_gpus):
    gpu_data = data[gpu_id * batch_size_per_gpu:(gpu_id + 1) * batch_size_per_gpu]
    outputs[gpu_id] = model(gpu_data)
```

### 2. 추론 최적화

#### 모델 양자화
```python
# 모델 양자화로 추론 속도 향상
import torch.quantization as quantization

# 동적 양자화
quantized_model = torch.quantization.quantize_dynamic(
    model, 
    {nn.Linear}, 
    dtype=torch.qint8
)

# 정적 양자화
model.eval()
quantized_model = torch.quantization.quantize(
    model,
    run_fn=calibration_fn,
    mapping=torch.quantization.get_default_qconfig_mapping()
)
```

#### ONNX 변환
```python
# ONNX 형식으로 변환하여 추론 최적화
import torch.onnx

# ONNX 모델 생성
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=11,
    do_constant_folding=True
)
```

## 문제 해결

### 1. 일반적인 문제

#### 메모리 부족
```python
# 배치 크기 감소
batch_size = 32  # 128에서 32로 감소

# 그래디언트 누적
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

#### 훈련 불안정
```python
# 그래디언트 클리핑
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 학습률 스케줄링
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=epochs
)
```

### 2. 성능 문제

#### 느린 추론
```python
# 모델 최적화
model.eval()
with torch.no_grad():
    # 추론 모드에서 실행
    output = model(input)
```

#### 낮은 정확도
```python
# 데이터 증강
from torchvision.transforms import Compose, RandomRotation, RandomCrop

transforms = Compose([
    RandomRotation(degrees=10),
    RandomCrop(size=224, padding=4)
])
```

## 모니터링 및 로깅

### 1. 훈련 모니터링

#### TensorBoard 로깅
```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/experiment_1')

for epoch in range(epochs):
    for batch_idx, batch in enumerate(dataloader):
        loss = model(batch)
        
        # 로그 기록
        writer.add_scalar('Loss/Train', loss, epoch * len(dataloader) + batch_idx)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
```

#### Weights & Biases 통합
```python
import wandb

# W&B 초기화
wandb.init(project="robovlms", name="experiment_1")

# 메트릭 로깅
wandb.log({
    "loss": loss,
    "accuracy": accuracy,
    "learning_rate": optimizer.param_groups[0]['lr']
})
```

### 2. 모델 성능 추적

#### 체크포인트 관리
```python
# 최고 성능 모델 저장
if val_accuracy > best_accuracy:
    best_accuracy = val_accuracy
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'accuracy': val_accuracy
    }, 'best_model.pt')
```

#### 성능 메트릭 계산
```python
# 성능 메트릭 계산
def calculate_metrics(predictions, targets):
    accuracy = (predictions == targets).float().mean()
    precision = calculate_precision(predictions, targets)
    recall = calculate_recall(predictions, targets)
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }
```
