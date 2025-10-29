# RoboVLMs Fine-tuning 코드 분석

## 1. 기본 Fine-tuning 스크립트

### 1.1 메인 학습 스크립트

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

### 1.2 학습 실행 스크립트

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
- **-nnodes**: 총 워커 노드 수 (분산 학습 규모)
- **-node_rank**: 현재 노드의 순위 (0부터 시작)
- **-nproc_per_node**: 노드당 프로세스 수 (GPU 수와 동일)
- **-master_addr**: 마스터 노드 IP 주소 (통신 담당)
- **-master_port**: 마스터 포트 번호 (6042, 통신 채널)
- **main.py**: 메인 학습 스크립트 실행
- **-exp_name ${subfix}**: 실험명 (타임스탬프 기반, 중복 방지)
- **${@:1}**: 스크립트에 전달된 모든 인자 (config 파일 등)
- **-gpus**: 사용할 GPU 수 (하드웨어 제약)
- **-num_nodes**: 총 노드 수 (분산 학습 규모)

### 1.3 CALVIN Fine-tuning 예시

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

## 2. DeepSpeed 통합 분석

### 2.1 DeepSpeed 체크포인트 변환

```python
# tools/convert_deepspeed_to_fp32.py
def convert_deepspeed_to_fp32(checkpoint_path, output_path):
    """
    DeepSpeed 체크포인트를 FP32로 변환
    """
    # DeepSpeed 체크포인트 로드
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # FP32로 변환
    fp32_checkpoint = {}
    for key, value in checkpoint.items():
        if isinstance(value, torch.Tensor):
            fp32_checkpoint[key] = value.float()
        else:
            fp32_checkpoint[key] = value
    
    # 변환된 체크포인트 저장
    torch.save(fp32_checkpoint, output_path)
```

### 2.2 메모리 효율성 최적화

```python
# 메모리 효율성을 위한 설정
{
    "precision": "bf16",              # Brain Float 16 (메모리 50% 절약)
    "gradient_checkpointing": true,   # Gradient Checkpointing 활성화
    "batch_size": 4,                  # 작은 배치 크기
    "gradient_accumulation_steps": 4, # 그래디언트 누적 (효과적 배치 크기 16)
    "max_grad_norm": 1.0              # 그래디언트 클리핑
}
```

## 3. 학습 파이프라인 분석

### 3.1 BaseTrainer 구조

```python
# robovlms/train/base_trainer.py
class BaseTrainer(pl.LightningModule):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.model = self._build_model()
        self.criterion = self._build_criterion()
        
    def training_step(self, batch, batch_idx):
        # Forward pass
        outputs = self.model(batch)
        
        # Loss 계산
        loss = self.criterion(outputs, batch['labels'])
        
        # 로깅
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.configs['learning_rate'],
            weight_decay=self.configs['weight_decay']
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.configs['max_epochs']
        )
        
        return [optimizer], [scheduler]
```

### 3.2 데이터 모듈 구조

```python
# robovlms/data/datamodule/gr_datamodule.py
class GRDataModule(pl.LightningDataModule):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = self._build_dataset('train')
            self.val_dataset = self._build_dataset('val')
            
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.configs['batch_size'],
            shuffle=True,
            num_workers=self.configs['num_workers']
        )
        
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.configs['batch_size'],
            shuffle=False,
            num_workers=self.configs['num_workers']
        )
```

## 4. 핵심 특징

### 4.1 Full Fine-tuning 방식
- **LoRA 비활성화**: 전체 모델 파라미터 학습
- **메모리 최적화**: Mixed precision, Gradient checkpointing
- **안정적 학습**: 낮은 학습률, 그래디언트 클리핑

### 4.2 분산 학습 지원
- **torchrun**: PyTorch 분산 학습 런처
- **DDP Strategy**: Data Parallel 분산 학습
- **체크포인트 변환**: DeepSpeed → FP32 호환성

### 4.3 실험 관리
- **타임스탬프**: 실험명에 타임스탬프 포함
- **로깅**: TensorBoard, CSV 로거
- **체크포인트**: 자동 저장 및 복원

## 5. 참고 자료

- `RoboVLMs/main.py`: 메인 학습 스크립트
- `RoboVLMs/scripts/run.sh`: 학습 실행 스크립트
- `RoboVLMs/robovlms/train/base_trainer.py`: 베이스 트레이너
- `RoboVLMs/robovlms/data/datamodule/gr_datamodule.py`: 데이터 모듈
- `RoboVLMs/tools/convert_deepspeed_to_fp32.py`: 체크포인트 변환 도구
