# Mobile VLA with RoboVLMs - 완전 가이드

## 개요

이 가이드는 RoboVLMs 프레임워크를 사용하여 Mobile VLA 모델을 학습하고 배포하는 전체 과정을 다룹니다.

## 목차

1. [태스크 정의](#태스크-정의)
2. [환경 설정](#환경-설정)
3. [데이터셋 준비](#데이터셋-준비)
4. [학습 실행](#학습-실행)
5. [추론 실행](#추론-실행)
6. [ROS2 통합](#ros2-통합)
7. [Docker 배포](#docker-배포)
8. [문제 해결](#문제-해결)

---

## 태스크 정의

### 🎯 **핵심 태스크: 모바일 로봇 네비게이션**

Mobile VLA의 핵심 태스크는 **모바일 로봇의 장애물 회피 네비게이션**입니다.

#### **8가지 시나리오**
- **1박스 시나리오**: 1개 박스 장애물을 피해 목표 지점 도달
- **2박스 시나리오**: 2개 박스 장애물을 피해 목표 지점 도달
- **경로 선택**: 각 시나리오마다 "left" 또는 "right" 경로 선택

#### **액션 공간 (2D)**
```python
action_space = {
    'linear_x': [-1.15, 1.15],    # 전진/후진 (m/s)
    'linear_y': [-1.15, 1.15],    # 좌우 이동 (m/s)
    'angular_z': [-1.15, 1.15],   # 회전 (rad/s)
    'action_type': [0, 3]         # 액션 타입 (0:이동, 1:회전, 2:정지, 3:특수)
}
```

#### **입력 데이터**
- **시각적**: RGB 카메라 이미지 (720×1280×3)
- **언어적**: 자연어 명령 ("Navigate around obstacles")
- **로봇 상태**: 위치, 속도, 방향 등 15차원 상태 벡터

자세한 태스크 정의는 [MOBILE_VLA_TASK_DEFINITION.md](./MOBILE_VLA_TASK_DEFINITION.md)를 참조하세요.

---

## 환경 설정

### 1. 필수 요구사항

- **하드웨어**:
  - GPU: NVIDIA GPU with 16GB+ VRAM (RTX 4090, A6000, etc.)
  - RAM: 32GB+
  - Storage: 100GB+ (for models and datasets)

- **소프트웨어**:
  - Ubuntu 22.04
  - CUDA 11.8+
  - Python 3.10+
  - ROS2 Humble (for inference)
  - Docker (optional, for containerized deployment)

### 2. RoboVLMs 설치

```bash
# Clone repository
cd /home/billy/25-1kp/vla
cd RoboVLMs

# Install dependencies
pip install -r requirements.txt

# Install RoboVLMs
pip install -e .
```

### 3. 추가 패키지 설치

```bash
# PyTorch Lightning
pip install pytorch-lightning

# Transformers
pip install transformers>=4.40.0

# Image processing
pip install pillow opencv-python

# ROS2 (if not installed)
sudo apt install ros-humble-desktop
sudo apt install python3-colcon-common-extensions
```

---

## 데이터셋 준비

### 1. Mobile VLA 데이터셋 구조

```
ROS_action/mobile_vla_dataset/
├── episode_20250101_120000_1box_vert_left_50cm.h5
├── episode_20250101_120100_1box_vert_right_50cm.h5
├── episode_20250101_120200_2box_hori_left_100cm.h5
└── ...
```

각 `.h5` 파일 구조:
```
/observations/images  : (T, H, W, 3) RGB images
/action              : (T, 4) actions [linear_x, linear_y, angular_z, action_type]
/episode_metadata    : scenario, distance, etc.
```

### 2. 데이터셋 통계 확인

```bash
python3 -c "
import h5py
import numpy as np
from pathlib import Path

data_dir = Path('../ROS_action/mobile_vla_dataset')
h5_files = list(data_dir.glob('*.h5'))

print(f'Total episodes: {len(h5_files)}')

total_frames = 0
for h5_path in h5_files:
    with h5py.File(h5_path, 'r') as f:
        total_frames += len(f['observations/images'])

print(f'Total frames: {total_frames}')
print(f'Average frames per episode: {total_frames / len(h5_files):.1f}')
"
```

---

## 학습 실행

### 1. 학습 설정 확인

설정 파일: `configs/mobile_vla/train_mobile_vla_full_ft.json`

주요 파라미터:
```json
{
  "task_name": "mobile_vla_full_finetune",
  "model": "kosmos",
  "window_size": 8,
  "fwd_pred_next_n": 1,
  "batch_size": 4,
  "learning_rate": 1e-5,
  "max_epochs": 50,
  "train_setup": {
    "freeze_backbone": false,
    "train_vision": true,
    "lora_enable": false,
    "gradient_checkpointing": true
  },
  "act_head": {
    "type": "LSTMDecoder",
    "num_layers": 4,
    "action_dim": 4,
    "action_space": "continuous"
  }
}
```

### 2. 테스트 모드 실행 (권장)

먼저 테스트 모드로 파이프라인이 정상 작동하는지 확인:

```bash
cd RoboVLMs
./scripts/run_mobile_vla_train.sh --test
```

예상 출력:
```
Found 132 .h5 files in dataset
Train dataset: 2100 samples
Val dataset: 276 samples
Model created: RoboKosMos_MobileVLA
Total parameters: 1,234,567,890
Trainable parameters: 1,234,567,890
Starting training...
Epoch 0: 100%|██████████| 10/10 [00:30<00:00]
```

### 3. 전체 학습 실행

```bash
./scripts/run_mobile_vla_train.sh
```

학습 모니터링:
```bash
# TensorBoard
tensorboard --logdir runs/mobile_vla/logs

# 실시간 로그 확인
tail -f runs/mobile_vla/logs/mobile_vla_full_finetune/version_0/events.out.tfevents.*
```

### 4. 체크포인트에서 재개

```bash
./scripts/run_mobile_vla_train.sh --resume runs/mobile_vla/checkpoints/mobile_vla-epoch=10-val_loss=0.1234.ckpt
```

### 5. 학습 결과 확인

```bash
ls -lh runs/mobile_vla/checkpoints/
```

예상 출력:
```
mobile_vla-epoch=10-val_loss=0.1234.ckpt  (6.9GB)
mobile_vla-epoch=20-val_loss=0.0987.ckpt  (6.9GB)
mobile_vla-last.ckpt                       (6.9GB)
```

---

## 추론 실행

### 1. 테스트 모드 (ROS2 없이)

```bash
./scripts/run_mobile_vla_inference.sh \
  --checkpoint runs/mobile_vla/checkpoints/mobile_vla-best.ckpt \
  --device cuda
```

예상 출력:
```
Mobile VLA Inference initialized
Device: cuda
Window size: 8
Running in test mode (no ROS2)
Predicted action: [0.5, -0.2, 0.1, 1.0]
Action chunk shape: (1, 4)
```

### 2. Python API 사용

```python
from eval.mobile_vla.inference_wrapper import MobileVLAInference
import numpy as np

# Initialize
inference = MobileVLAInference(
    checkpoint_path='runs/mobile_vla/checkpoints/mobile_vla-best.ckpt',
    config_path='configs/mobile_vla/train_mobile_vla_full_ft.json',
    device='cuda'
)

# Predict
image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
instruction = "Navigate around obstacles"

result = inference.predict(image, instruction)

print(f"Action: {result['action']}")
# Output: Action: [0.5, -0.2, 0.1, 1.0]
```

---

## ROS2 통합

### 1. ROS2 환경 설정

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
```

### 2. ROS2 노드 실행

```bash
./scripts/run_mobile_vla_inference.sh \
  --checkpoint runs/mobile_vla/checkpoints/mobile_vla-best.ckpt \
  --ros2
```

### 3. ROS2 토픽 확인

```bash
# 다른 터미널에서
ros2 topic list
```

예상 출력:
```
/camera/image_raw
/vla_text_command
/cmd_vel
```

### 4. 테스트 명령 전송

```bash
# 카메라 이미지 퍼블리시 (예시)
ros2 run image_publisher image_publisher_node /path/to/image.jpg

# 텍스트 명령 전송
ros2 topic pub /vla_text_command std_msgs/msg/String \
  "data: 'Navigate around obstacles'"

# cmd_vel 확인
ros2 topic echo /cmd_vel
```

---

## Docker 배포

### 1. Docker 이미지 빌드

```bash
cd RoboVLMs
docker build -t robovlms-mobile-vla:latest .
```

### 2. Docker Compose로 학습 실행

```bash
docker-compose -f docker-compose-mobile-vla.yml up train_mobile_vla
```

### 3. Docker Compose로 추론 실행

```bash
docker-compose -f docker-compose-mobile-vla.yml up inference_mobile_vla
```

### 4. 테스트 모드

```bash
docker-compose -f docker-compose-mobile-vla.yml up test_mobile_vla
```

---

## 문제 해결

### 1. CUDA Out of Memory

**증상**: `RuntimeError: CUDA out of memory`

**해결책**:
```json
// configs/mobile_vla/train_mobile_vla_full_ft.json
{
  "batch_size": 2,  // 4 → 2로 감소
  "train_setup": {
    "gradient_checkpointing": true,
    "precision": "bf16-mixed"
  },
  "trainer": {
    "accumulate_grad_batches": 2  // Effective batch size = 2 * 2 = 4
  }
}
```

### 2. 데이터셋 로딩 실패

**증상**: `No .h5 files found in dataset`

**해결책**:
```bash
# 데이터 경로 확인
ls -la ../ROS_action/mobile_vla_dataset/*.h5

# 설정 파일 수정
vim configs/mobile_vla/train_mobile_vla_full_ft.json
# "data_dir": "/absolute/path/to/mobile_vla_dataset"
```

### 3. ROS2 토픽 연결 안됨

**증상**: `No publishers on /camera/image_raw`

**해결책**:
```bash
# ROS_DOMAIN_ID 확인
echo $ROS_DOMAIN_ID

# 네트워크 모드 확인
docker-compose -f docker-compose-mobile-vla.yml config | grep network_mode
# network_mode: host 확인

# DDS 설정 확인
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
```

### 4. 모델 성능 낮음

**증상**: Validation loss가 감소하지 않음

**해결책**:
1. 학습률 조정:
   ```json
   "learning_rate": 5e-6  // 1e-5 → 5e-6
   ```

2. Warmup 추가:
   ```json
   "warmup_epochs": 5
   ```

3. 데이터 증강 활성화:
   ```json
   "train_dataset": {
     "augmentation": {
       "color_jitter": 0.2,
       "gaussian_noise": 0.02
     }
   }
   ```

---

## 성능 벤치마크

### 예상 학습 시간 (RTX 4090)

| Epochs | Batch Size | Time     | Final Val Loss |
|--------|-----------|----------|----------------|
| 10     | 4         | ~2 hours | 0.15           |
| 50     | 4         | ~10 hours| 0.08           |

### 추론 속도

| Device       | Latency (ms) | FPS  |
|--------------|--------------|------|
| RTX 4090     | 50           | 20   |
| Jetson Orin  | 150          | 6.7  |

---

## 다음 단계

1. **하이퍼파라미터 튜닝**: Learning rate, batch size, window size 최적화
2. **데이터 증강**: 더 많은 시나리오 수집
3. **모델 압축**: Quantization, Pruning으로 Jetson 배포 최적화
4. **실제 로봇 테스트**: 시뮬레이션 → 실제 환경 전환

---

## 참고 자료

- [RoboVLMs 논문](https://arxiv.org/abs/2412.04139)
- [RoboVLMs GitHub](https://github.com/RoboVLMs/RoboVLMs)
- [PyTorch Lightning 문서](https://lightning.ai/docs/pytorch/stable/)
- [ROS2 Humble 문서](https://docs.ros.org/en/humble/)

---

## 라이센스

이 프로젝트는 RoboVLMs 라이센스를 따릅니다.

