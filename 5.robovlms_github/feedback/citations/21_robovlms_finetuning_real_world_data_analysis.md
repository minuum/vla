# 21. RoboVLMs Fine-tuning 및 Real-World 데이터 수집 분석

## 1. RoboVLMs Fine-tuning 코드 분석

### **1.1 기본 Fine-tuning 스크립트**

**메인 학습 스크립트**: `RoboVLMs/main.py`
```python
# RoboVLMs/main.py:1-100
import os                    # 운영체제 인터페이스 (파일 경로, 환경변수 등)
import argparse             # 명령행 인자 파싱 (config 파일 경로, 하이퍼파라미터 등)
import json                 # JSON 설정 파일 읽기/쓰기
from pathlib import Path    # 경로 객체 처리 (크로스 플랫폼 호환성)
import importlib            # 동적 모듈 임포트 (설정에 따른 모델/데이터셋 로딩)
import copy                 # 깊은 복사 (설정 딕셔너리 복제)
import functools            # 함수 데코레이터 (캐싱, 부분 적용 등)
from re import L            # 정규표현식 (문자열 패턴 매칭)
from typing import Dict, Any # 타입 힌트 (코드 가독성 및 IDE 지원)
import datetime             # 날짜/시간 처리 (실험명 생성, 로그 타임스탬프)

# PyTorch Lightning 관련 임포트
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint  # 학습률 모니터링, 체크포인트 저장
from lightning.pytorch.trainer import Trainer                                  # 메인 학습기
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger            # 로깅 (텐서보드, CSV)
from lightning.pytorch.strategies import DDPStrategy                          # 분산 학습 전략
from lightning import seed_everything                                         # 재현 가능한 랜덤 시드
import torch                                                                  # PyTorch 핵심
import torch.distributed as dist                                              # 분산 학습 통신

# RoboVLMs 커스텀 모듈들
from robovlms.train.base_trainer import BaseTrainer                          # VLA 학습기 (LSTM + VLM 통합)
from robovlms.data.datamodule.gr_datamodule import GRDataModule              # 데이터 로더 (CALVIN, OXE 등)
from robovlms.data.data_utils import preprocess_image                        # 이미지 전처리 (CLIP 정규화)
from robovlms.utils.setup_callback import SetupCallback                      # 실험 설정 콜백
```

**출처**: `RoboVLMs/main.py:1-25`

**각 라인 상세 설명**:
- **os**: 파일 시스템 접근, 환경변수 설정 (CUDA_VISIBLE_DEVICES 등)
- **argparse**: 커맨드라인에서 config 파일 경로, GPU 수, 학습률 등 받기
- **json**: 학습 설정 파일 (configs/*.json) 읽기
- **pathlib.Path**: 크로스 플랫폼 경로 처리 (Windows/Linux 호환성)
- **importlib**: 설정에 따라 다른 모델/데이터셋 동적 로딩
- **copy**: 설정 딕셔너리 깊은 복사 (원본 보존)
- **functools**: 함수 캐싱, 부분 적용 등 고급 함수 처리
- **typing**: 타입 힌트로 코드 가독성 향상
- **datetime**: 실험명에 타임스탬프 포함 (중복 방지)
- **Lightning**: PyTorch Lightning 프레임워크 (분산 학습, 로깅, 체크포인트 자동화)
- **BaseTrainer**: RoboVLMs 커스텀 학습기 (VLM + LSTM 통합 학습)
- **GRDataModule**: 데이터 로딩 및 전처리 (CALVIN, OXE, Custom Dataset)
- **preprocess_image**: CLIP 표준 이미지 전처리 (정규화, 리사이징)
- **SetupCallback**: 실험 설정 저장 및 로깅

### **1.2 학습 실행 스크립트**

**실행 스크립트**: `RoboVLMs/scripts/run.sh`
```bash
#!/usr/bin/env bash                    # Bash 스크립트 선언

conda activate robovlm                 # Conda 환경 활성화 (Python 패키지 관리)

# 분산 학습 설정
GPUS_PER_NODE=1                        # 노드당 GPU 수 (단일 GPU 학습)
WORKER_NUM=1                           # 워커 노드 수 (단일 머신)
NODE_ID=0                              # 현재 노드 ID (마스터 노드)
METIS_WORKER_0_HOST=127.0.0.1         # 마스터 노드 IP (로컬호스트)

# DeepSpeed 체크포인트 변환
if [ $NODE_ID == "0" ]; then           # 마스터 노드에서만 실행
  echo "---------- Converting deepspeed checkpoint to fp32. ----------"
  python3 tools/convert_deepspeed_to_fp32.py ${@:1}  # DeepSpeed → FP32 변환
fi

# 학습 실행
torchrun \                             # PyTorch 분산 학습 런처
    --nnodes $WORKER_NUM \             # 총 노드 수
    --node_rank $NODE_ID \             # 현재 노드 순위
    --nproc_per_node $GPUS_PER_NODE \  # 노드당 프로세스 수
    --master_addr $METIS_WORKER_0_HOST \ # 마스터 노드 주소
    --master_port $port \              # 마스터 포트 (6042)
    main.py \                          # 메인 학습 스크립트
    --exp_name ${subfix} \             # 실험명 (타임스탬프)
    ${@:1} \                           # 전달받은 모든 인자
    --gpus $GPUS_PER_NODE \            # GPU 수
    --num_nodes $WORKER_NUM            # 노드 수
```

**출처**: `RoboVLMs/scripts/run.sh:1-58`

**각 라인 상세 설명**:
- **#!/usr/bin/env bash**: Bash 인터프리터 지정 (크로스 플랫폼 호환성)
- **conda activate robovlm**: RoboVLMs 전용 Python 환경 활성화
- **GPUS_PER_NODE=1**: 단일 GPU 학습 (메모리 효율성)
- **WORKER_NUM=1**: 단일 머신 학습 (분산 학습 비활성화)
- **NODE_ID=0**: 마스터 노드 (분산 학습 시 통신 담당)
- **METIS_WORKER_0_HOST=127.0.0.1**: 로컬호스트 (단일 머신)
- **if [ $NODE_ID == "0" ]**: 마스터 노드에서만 체크포인트 변환
- **echo**: 진행 상황 출력 (사용자 피드백)
- **python3 tools/convert_deepspeed_to_fp32.py**: DeepSpeed 체크포인트를 FP32로 변환 (호환성)
- **torchrun**: PyTorch 분산 학습 런처 (멀티 GPU/노드 지원)
- **--nnodes**: 총 워커 노드 수 (분산 학습 규모)
- **--node_rank**: 현재 노드의 순위 (0부터 시작)
- **--nproc_per_node**: 노드당 프로세스 수 (GPU 수와 동일)
- **--master_addr**: 마스터 노드 IP 주소 (통신 담당)
- **--master_port**: 마스터 포트 번호 (6042, 통신 채널)
- **main.py**: 메인 학습 스크립트 실행
- **--exp_name ${subfix}**: 실험명 (타임스탬프 기반, 중복 방지)
- **${@:1}**: 스크립트에 전달된 모든 인자 (config 파일 등)
- **--gpus**: 사용할 GPU 수 (하드웨어 제약)
- **--num_nodes**: 총 노드 수 (분산 학습 규모)

### **1.3 CALVIN Fine-tuning 예시**

**CALVIN Fine-tuning 명령어**:
```bash
bash scripts/run.sh configs/calvin_finetune/finetune_kosmos_cont-lstm-post_full-ft_text_vision_wd-0_ws-8_act-10.json
```

**출처**: `RoboVLMs/README.md:289-292`

**명령어 상세 분석**:
- **bash scripts/run.sh**: 앞서 분석한 학습 실행 스크립트 호출
- **configs/calvin_finetune/**: CALVIN 데이터셋 Fine-tuning 설정 디렉토리
- **finetune_kosmos**: Kosmos VLM 백본 사용
- **cont-lstm-post**: Continuous action space + LSTM policy head + post-processing
- **full-ft**: Full Fine-tuning (LoRA 비활성화)
- **text_vision**: Text와 Vision 모두 학습
- **wd-0**: Weight decay 0 (정규화 비활성화)
- **ws-8**: Window size 8 (히스토리 길이)
- **act-10**: Action chunk size 10 (예측할 액션 수)
- **.json**: JSON 설정 파일 형식

---

## 2. Real-World 데이터 수집 방법

### **2.1 RoboVLMs가 사용하는 Real-World 데이터**

**CALVIN 데이터셋 (Real-World 특성)**:
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

**Real-World 데이터 구성**:
- **Franka Emika Panda 7-DOF 로봇팔**: 실제 로봇 하드웨어
- **다중 카메라 시스템**: 정적 카메라 + 그리퍼 카메라
- **실제 물리 환경**: 테이블, 물체, 조작 공간
- **다양한 태스크**: pick-and-place, navigation, manipulation
- **실제 로봇 조작**: 전문가가 직접 조작하여 데이터 수집

### **2.2 Custom Dataset 지원**

**Custom Dataset 형식**:
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

### **2.3 Custom Dataset 설정 예시**

**실제 RoboVLMs Custom Dataset Config 예시**:
```python
# CALVIN 데이터셋 설정 (실제 경로)
"train_dataset": {
    "type": "DiskCalvinDataset",                    # CALVIN 데이터셋 클래스
    "data_dir": "calvin/dataset/task_ABCD_D/training", # CALVIN 학습 데이터 경로
    "shift_first": false,                           # 첫 번째 프레임 시프트 비활성화
    "model_name": "kosmos",                        # Kosmos VLM 모델명
    "rgb_pad": 10,                                 # RGB 이미지 랜덤 시프트 크기 (픽셀)
    "gripper_pad": 4,                              # 그리퍼 이미지 랜덤 시프트 크기 (픽셀)
    "few_shot": true                               # Few-shot 학습 활성화
},
"val_dataset": {
    "type": "DiskCalvinDataset",                   # 동일한 CALVIN 데이터셋 클래스
    "data_dir": "calvin/dataset/task_ABCD_D/validation", # CALVIN 검증 데이터 경로
    "model_name": "kosmos"                          # 동일한 Kosmos 모델명
}
```

**OpenVLA 데이터셋 설정 (실제 경로)**:
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

**K-프로젝트 로봇카 네비게이션 데이터셋 설정 (실제 경로)**:
```python
"dataset": {                                      # K-프로젝트 학습 데이터셋
    "data_dir": "data/k_project/automotive/train", # K-프로젝트 학습 데이터 경로
    "data_name_list": ["carla_automotive", "real_navigation", "calvin_converted"], # 데이터셋 목록
    "task_name": "k_project_automotive_navigation", # 태스크명
    "batch_size": 4,                              # 배치 크기
    "seq_len": 9,                                 # 시퀀스 길이
    "window_size": 8,                             # 윈도우 크기
    "chunk_size": 1,                              # 청크 크기
    "image_size": 224,                            # 이미지 크기 (CLIP 표준)
    "data_type": "ros2_calvin",                   # ROS2-CALVIN 변환 데이터 타입
    "conversion_mode": "automotive",               # 자동차 네비게이션 변환 모드
    "normalize_actions": true,                    # 액션 정규화 활성화
    "action_dim": 4,                              # 4차원 액션 (linear_x, linear_y, angular_z, action_type)
    "save_format": "npz",                         # NumPy 압축 형식으로 저장
    "augmentation": {                             # 데이터 증강 설정
        "random_crop": false,                     # 랜덤 크롭 비활성화
        "random_flip": false,                     # 랜덤 플립 비활성화
        "color_jitter": 0.1,                      # 색상 지터링 10%
        "gaussian_noise": 0.01                    # 가우시안 노이즈 1%
    }
},
"val_dataset": {                                 # K-프로젝트 검증 데이터셋
    "data_dir": "data/k_project/automotive/val",  # K-프로젝트 검증 데이터 경로
    "data_name_list": ["carla_val", "vehicle_command_val", "sequential_task_val"], # 검증 데이터셋 목록
    "task_name": "k_project_automotive_val",     # 검증 태스크명
    "batch_size": 2,                             # 검증 배치 크기 (학습의 절반)
    "seq_len": 9,                                # 동일한 시퀀스 길이
    "window_size": 8,                             # 동일한 윈도우 크기
    "chunk_size": 1,                             # 동일한 청크 크기
    "image_size": 224,                           # 동일한 이미지 크기
    "data_type": "ros2_action",                  # ROS2 액션 데이터 타입
    "normalize_actions": true,                   # 동일한 액션 정규화
    "action_dim": 4                              # 동일한 4차원 액션
}
```

**출처**: 
- `RoboVLMs/README.md:323-382` (CALVIN, OpenVLA 설정)
- `RoboVLMs/configs/k_project/ros2_automotive.json:103-137` (K-프로젝트 설정)

**각 설정 상세 설명**:

**CALVIN 데이터셋 설정**:
- **"type": "DiskCalvinDataset"**: CALVIN 데이터셋 전용 클래스
- **"data_dir": "calvin/dataset/task_ABCD_D/training"**: CALVIN 공식 데이터 경로
- **"shift_first": false**: 첫 번째 프레임 시프트 비활성화 (데이터 일관성)
- **"model_name": "kosmos"**: Kosmos-2 VLM 모델명 (토크나이저 매칭)
- **"rgb_pad": 10**: RGB 이미지 랜덤 시프트 10픽셀 (데이터 증강)
- **"gripper_pad": 4**: 그리퍼 이미지 랜덤 시프트 4픽셀 (세밀한 증강)
- **"few_shot": true**: Few-shot 학습 활성화 (제한된 데이터 활용)

**OpenVLA 데이터셋 설정**:
- **"type": "OpenVLADataset"**: Open X-Embodiment 데이터셋 클래스
- **"data_root_dir": "openvla/datasets/open-x-embodiment"**: OpenVLA 공식 데이터 루트
- **"data_mix": "bridge"**: Bridge 데이터 믹스 (다양한 로봇 데이터 통합)
- **"window_sample": "sliding"**: 슬라이딩 윈도우 샘플링 (연속성 유지)
- **"organize_type": "interleave"**: 인터리브 데이터 구성 (다양한 태스크 혼합)
- **"shuffle_buffer_size": 51200**: 대용량 셔플 버퍼 (데이터 다양성)

**K-프로젝트 로봇카 네비게이션 설정**:
- **"data_dir": "data/k_project/automotive/train"**: K-프로젝트 전용 데이터 경로
- **"data_name_list"**: CARLA 시뮬레이션, 실제 네비게이션, CALVIN 변환 데이터
- **"data_type": "ros2_calvin"**: ROS2-CALVIN 변환 데이터 타입
- **"conversion_mode": "automotive"**: 로봇팔→로봇카 도메인 적응
- **"action_dim": 4**: 4차원 액션 공간 (linear_x, linear_y, angular_z, action_type)
- **"augmentation"**: 자동차 환경에 특화된 데이터 증강 설정

---

## 3. 우리 태스크에서의 Fine-tuning 방법

### **3.1 데이터 수집 방법**

**Real-World 데이터 수집을 위한 설정**:

1. **로봇 하드웨어**: 7-DOF 로봇팔 (Franka Emika Panda 또는 유사)
2. **카메라 시스템**: 
   - 정적 카메라 (rgb_static)
   - 그리퍼 카메라 (rgb_gripper)
3. **데이터 형식**: HDF5 또는 JSON 형태로 저장
4. **언어 명령**: 각 에피소드마다 태스크 설명

### **3.2 데이터 전처리**

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

**각 전처리 단계 상세 설명**:
- **이미지 전처리**:
  - **image_mean/std**: CLIP 모델의 ImageNet 사전학습 정규화 파라미터
  - **Resize((224, 224))**: CLIP 입력 크기 (224x224 픽셀)
  - **RandomHorizontalFlip(p=0.1)**: 10% 확률로 수평 뒤집기 (데이터 증강)
  - **ColorJitter**: 밝기와 대비를 10% 범위에서 랜덤 조정
  - **Normalize**: CLIP 표준으로 정규화 (모델 호환성)
- **액션 정규화**:
  - **norm_min/max**: CALVIN 데이터셋의 안전한 액션 범위
  - **arm_action**: 6-DOF 로봇 팔 액션 (위치 3 + 회전 3)
  - **np.clip**: 액션을 안전 범위로 제한 (로봇 안전성)
  - **gripper_action**: 그리퍼 액션을 [0, 1] 범위로 변환 (이진 분류)

### **3.3 Fine-tuning 설정**

**학습 설정**:
```json
{
    "train_setup": {                     // 학습 설정 그룹
        "precision": "bf16",             // Mixed precision (메모리 절약)
        "freeze_backbone": false,        // 전체 VLM 백본 학습 (Full Fine-tuning)
        "train_vision": true,            // Vision encoder 학습 (CLIP)
        "freeze_resampler": false,       // Vision resampler 학습 (멀티모달 융합)
        "train_text_embedding": true,    // Text embedding 학습 (언어 이해)
        "lora_enable": false,            // LoRA 비활성화 (Full Fine-tuning)
        "train_full_decoder": false,     // 전체 decoder는 비활성화 (선택적 학습)
        "train_decoder_layers": -1      // 특정 레이어 수 제한 없음 (전체 학습)
    },
    "learning_rate": 2e-5,              // 낮은 학습률 (안정적 학습)
    "min_lr_scale": 1e-2,               // 최소 학습률 스케일 (10%)
    "weight_decay": 0,                  // Weight decay 비활성화 (정규화 없음)
    "warmup_epochs": 0.25,              // Warmup 에포크 (학습률 스케줄링)
    "batch_size": 4,                    // 작은 배치 크기 (메모리 효율성)
    "max_epochs": 5,                    // 짧은 학습 기간 (빠른 수렴)
    "gradient_clip_val": 1.0            // 그래디언트 클리핑 (학습 안정성)
}
```

**각 설정 상세 설명**:
- **"train_setup"**: 학습 관련 설정 그룹
  - **"precision": "bf16"**: Brain Float 16 정밀도 (메모리 50% 절약)
  - **"freeze_backbone": false**: VLM 백본 전체 학습 (Full Fine-tuning)
  - **"train_vision": true**: CLIP Vision Encoder 학습 (비전 이해)
  - **"freeze_resampler": false**: Vision Resampler 학습 (멀티모달 융합)
  - **"train_text_embedding": true**: Text Embedding 학습 (언어 이해)
  - **"lora_enable": false**: LoRA 비활성화 (Full Fine-tuning 우선)
  - **"train_full_decoder": false**: 전체 Decoder 학습 비활성화 (선택적)
  - **"train_decoder_layers": -1**: 특정 레이어 수 제한 없음 (전체 학습)
- **학습 하이퍼파라미터**:
  - **"learning_rate": 2e-5**: 낮은 학습률 (안정적 수렴)
  - **"min_lr_scale": 1e-2**: 최소 학습률 (10%까지 감소)
  - **"weight_decay": 0**: 정규화 비활성화 (과적합 방지)
  - **"warmup_epochs": 0.25**: 학습률 점진적 증가 (안정적 시작)
  - **"batch_size": 4**: 작은 배치 크기 (메모리 효율성)
  - **"max_epochs": 5**: 짧은 학습 기간 (빠른 수렴)
  - **"gradient_clip_val": 1.0**: 그래디언트 클리핑 (폭발 방지)

### **3.4 학습 실행**

**학습 명령어**:
```bash
# 1. 환경 활성화
conda activate robovlm                 # RoboVLMs Python 환경 활성화

# 2. 학습 실행
bash scripts/run.sh configs/custom_finetune/finetune_kosmos_cont-lstm-post_full-ft_text_vision_wd-0_ws-8_act-10.json
```

**학습 파라미터**:
- **Window Size**: 8 (히스토리 길이) - 과거 8개 프레임 사용
- **Action Chunk Size**: 10 (예측할 액션 수) - 미래 10개 액션 예측
- **Batch Size**: 4 (메모리 효율성) - GPU 메모리 제한
- **Learning Rate**: 2e-5 (낮은 학습률) - 안정적 수렴
- **Max Epochs**: 5 (짧은 학습 기간) - 빠른 수렴

**명령어 상세 분석**:
- **conda activate robovlm**: RoboVLMs 전용 Python 환경 활성화
- **bash scripts/run.sh**: 앞서 분석한 학습 실행 스크립트 호출
- **configs/custom_finetune/**: 커스텀 Fine-tuning 설정 디렉토리
- **finetune_kosmos**: Kosmos VLM 백본 사용
- **cont-lstm-post**: Continuous action + LSTM policy head + post-processing
- **full-ft**: Full Fine-tuning (LoRA 비활성화)
- **text_vision**: Text와 Vision 모두 학습
- **wd-0**: Weight decay 0 (정규화 비활성화)
- **ws-8**: Window size 8 (히스토리 길이)
- **act-10**: Action chunk size 10 (예측할 액션 수)
- **.json**: JSON 설정 파일 형식

---

## 4. 참고할 코드와 수정 사항

### **4.1 참고할 핵심 코드**

1. **학습 스크립트**: `RoboVLMs/scripts/run.sh`
2. **메인 학습 코드**: `RoboVLMs/main.py`
3. **데이터 모듈**: `RoboVLMs/robovlms/data/datamodule/gr_datamodule.py`
4. **베이스 트레이너**: `RoboVLMs/robovlms/train/base_trainer.py`
5. **데이터셋**: `RoboVLMs/robovlms/data/calvin_dataset.py`

### **4.2 수정해야 할 부분**

**1. Custom Dataset 클래스 생성**:
```python
# robovlms/data/custom_dataset.py
class CustomDataset(ActionPredictionDataset):  # RoboVLMs 기본 데이터셋 상속
    def __init__(self, data_dir, **kwargs):     # 데이터 디렉토리와 설정 받기
        # Custom 데이터 로딩 로직 구현
        self.data_dir = data_dir                # 데이터 경로 저장
        self.episodes = self._load_episodes()   # 에피소드 목록 로드
        super().__init__(**kwargs)             # 부모 클래스 초기화
    
    def _load_episodes(self):                  # 에피소드 로딩 함수
        # HDF5/JSON 파일에서 에피소드 목록 로드
        pass
    
    def __getitem__(self, idx):                # 데이터 아이템 반환
        # Custom 데이터 반환 로직 구현
        episode = self.episodes[idx]            # 에피소드 선택
        images = self._load_images(episode)    # 이미지 로드
        actions = self._load_actions(episode)  # 액션 로드
        language = self._load_language(episode) # 언어 명령 로드
        return self._process_sample(images, actions, language)  # 전처리 후 반환
```

**2. Config 파일 생성**:
```json
// configs/custom_finetune/finetune_kosmos_custom.json
{
    "train_dataset": {                         // 학습 데이터셋 설정
        "type": "CustomDataset",               // 커스텀 데이터셋 클래스
        "data_dir": "path/to/your/custom_data", // 데이터 디렉토리 경로
        "model_name": "kosmos",                // VLM 모델명 (토크나이저 매칭)
        "window_size": 8,                      // 히스토리 윈도우 크기
        "fwd_pred_next_n": 10                  // 예측할 액션 수
    },
    "train_setup": {                           // 학습 설정
        "freeze_backbone": false,              // VLM 백본 학습 (Full Fine-tuning)
        "train_vision": true,                  // Vision encoder 학습
        "lora_enable": false                   // LoRA 비활성화
    }
}
```

**3. 데이터 전처리 함수**:
```python
# robovlms/data/data_utils.py에 추가
def preprocess_custom_data(images, actions, language):
    # Custom 데이터 전처리 로직
    # 1. 이미지 전처리 (CLIP 표준)
    processed_images = preprocess_image(images)  # CLIP 정규화 적용
    
    # 2. 액션 정규화 (CALVIN 표준)
    normalized_actions = normalize_action(actions)  # [-0.65, 0.65] 범위로 정규화
    
    # 3. 언어 토큰화 (Kosmos 토크나이저)
    tokenized_language = tokenize_language(language)  # 자연어 → 토큰 ID
    
    return processed_images, normalized_actions, tokenized_language
```

**각 구현 상세 설명**:
- **CustomDataset 클래스**:
  - **ActionPredictionDataset 상속**: RoboVLMs 기본 데이터셋 기능 활용
  - **__init__**: 데이터 디렉토리와 설정 파라미터 받기
  - **_load_episodes**: HDF5/JSON 파일에서 에피소드 목록 로드
  - **__getitem__**: 인덱스에 해당하는 데이터 샘플 반환
  - **_load_images/actions/language**: 각 데이터 타입별 로딩 함수
  - **_process_sample**: 데이터 전처리 및 변환
- **Config 파일**:
  - **"train_dataset"**: 학습용 데이터셋 설정
  - **"type": "CustomDataset"**: 커스텀 데이터셋 클래스 지정
  - **"data_dir"**: 데이터 파일들이 저장된 디렉토리 경로
  - **"model_name": "kosmos"**: VLM 모델명 (토크나이저 매칭)
  - **"window_size": 8**: 히스토리 윈도우 크기 (과거 8개 프레임)
  - **"fwd_pred_next_n": 10**: 예측할 액션 수 (미래 10개 액션)
  - **"train_setup"**: 학습 관련 설정
  - **"freeze_backbone": false**: VLM 백본 전체 학습
  - **"train_vision": true**: Vision encoder 학습
  - **"lora_enable": false**: LoRA 비활성화 (Full Fine-tuning)
- **데이터 전처리 함수**:
  - **preprocess_image**: CLIP 표준 이미지 전처리 (정규화, 리사이징)
  - **normalize_action**: CALVIN 표준 액션 정규화 ([-0.65, 0.65] 범위)
  - **tokenize_language**: Kosmos 토크나이저로 언어 토큰화

### **4.3 학습 실행 순서**

1. **데이터 준비**: Custom 데이터셋을 RoboVLMs 형식으로 변환
   - HDF5/JSON 파일을 RoboVLMs 형식으로 변환
   - 이미지, 액션, 언어 명령 데이터 구조화
   - 학습/검증 데이터 분할

2. **Config 설정**: 학습 설정 파일 생성
   - `configs/custom_finetune/finetune_kosmos_custom.json` 생성
   - 데이터셋 경로, 모델 설정, 하이퍼파라미터 지정
   - Full Fine-tuning 설정 (LoRA 비활성화)

3. **환경 설정**: Conda 환경 및 의존성 설치
   - `conda activate robovlm` 환경 활성화
   - PyTorch, Lightning, RoboVLMs 의존성 확인
   - GPU 메모리 및 CUDA 설정 확인

4. **학습 실행**: `bash scripts/run.sh config_file.json`
   - 학습 스크립트 실행
   - 실시간 로그 모니터링
   - 체크포인트 자동 저장

5. **모델 평가**: 학습된 모델 성능 평가
   - CALVIN 벤치마크와 유사한 평가 방법 사용
   - 성공률, 정확도, 안정성 지표 측정
   - 실시간 로봇 제어 테스트

---

## 5. 핵심 결론

### **5.1 RoboVLMs Fine-tuning 방법**

- **Full Fine-tuning**: LoRA 대신 전체 모델 학습
- **On-Device 최적화**: 메모리 효율성을 위한 설정
- **Custom Dataset 지원**: 자체 데이터셋으로 Fine-tuning 가능

### **5.2 Real-World 데이터 수집**

- **CALVIN 데이터셋**: 실제 로봇 하드웨어로 수집된 데이터
- **Custom Dataset**: 자체 로봇 환경에서 데이터 수집 가능
- **데이터 형식**: HDF5/JSON 형태로 저장

### **5.3 우리 태스크 적용 방안**

1. **데이터 수집**: 7-DOF 로봇팔 + 다중 카메라 시스템
2. **데이터 전처리**: RoboVLMs 형식으로 변환
3. **Fine-tuning**: Full Fine-tuning 방식 적용
4. **평가**: CALVIN 벤치마크와 유사한 평가 방법 사용

**출처 요약**:
- `RoboVLMs/scripts/run.sh`: 학습 실행 스크립트
- `RoboVLMs/main.py`: 메인 학습 코드
- `RoboVLMs/README.md:300-382`: Custom Dataset 설정
- `RoboVLMs/robovlms/data/calvin_dataset.py`: 데이터셋 구현
- `RoboVLMs/robovlms/train/base_trainer.py`: 학습 로직

**핵심**: RoboVLMs는 **Full Fine-tuning 방식**을 사용하며, **Custom Dataset을 통한 Real-World 데이터 수집**이 가능합니다.
