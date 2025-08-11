## Mobile VLA 개발·운영 계획 (요약)

### 디렉토리 구조 파악
- `data/`: HDF5 데이터셋 로더 (`MobileVLADataset`)
- `models/`: 인코더(이미지/텍스트), 멀티모달 융합, 정책 헤드
- `training/`: Lightning 트레이너와 간이(PyTorch) 트레이너
- `README.md`: 개요/사용법

### 실행 계획(의사코드)
1) 환경/의존성
```bash
# requirements.txt (예시)
torch
torchvision
transformers
h5py
numpy
pillow
lightning
tqdm

# 데이터 경로 환경 변수(예시)
export MOBILE_VLA_DATA_DIR=/Users/minu/dev/vla/ROS_action/mobile_vla_dataset
```

2) 스모크 테스트
```python
# Dataset
from Robo+/Mobile_VLA/data import MobileVLADataset
ds = MobileVLADataset(data_dir=os.getenv("MOBILE_VLA_DATA_DIR"))
assert len(ds) >= 0
sample = ds[0] if len(ds) else None

# Model
from Robo+/Mobile_VLA/models import MobileVLAModel
images = torch.randn(2, 18, 3, 224, 224)
scenarios = ["1box_vert_left", "2box_hori_right"]
model = MobileVLAModel(hidden_size=768, use_lite_mode=False)
out = model(images, scenarios)

# Simple Trainer one step
from Robo+/Mobile_VLA/training.mobile_trainer_simple import SimpleMobileVLATrainer
cfg = {"hidden_size": 512, "use_lite_mode": True, "batch_size": 2, "sequence_length": 18}
trainer = SimpleMobileVLATrainer(cfg)
# batch = 실제 소규모 배치 or 더미 데이터로 검증
```

3) 설정 표준화
- `configs/mobile_vla_default.json` 추가 후 환경변수 참조
- `training/train_mobile_vla.py` 런처: CLI 인자 > JSON > 환경변수 순으로 오버라이드

4) 로깅/체크포인트
- Lightning: TensorBoardLogger, ModelCheckpoint(`Robo+/Mobile_VLA/experiments/...`)
- Simple: tqdm + 주기적 `torch.save(...)`

5) 추론/운영 연계
- `MobileVLAModel.get_mobile_vla_action()` 사용해 실시간 추론
- 선택: ROS2 노드(`mobile_vla_infer`) 생성해 `/camera/image_raw` 구독, `/mobile_vla/action` 퍼블리시

6) 배포/최적화(선택)
- Jetson: `use_lite_mode=True`, FP16, TorchScript/ONNX export

### 즉시 작업 목록(제안)
- `requirements.txt` 생성 및 문서화
- `configs/mobile_vla_default.json` + `training/train_mobile_vla.py` 추가
- `README.md`에 환경 변수/실행 가이드 보강
- 스모크 테스트 3종 실행(데이터셋/모델/트레이너)


