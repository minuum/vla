import torch
from pathlib import Path
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from transformers import BitsAndBytesConfig

# 로컬 모듈 임포트 (경로 설정 필요할 수 있음)
import sys
sys.path.append("/home/billy/25-1kp/vla/RoboVLMs")

from robovlms.model.policy_head.mobile_vla_policy import MobileVLATrainer
from robovlms.data.mobile_vla_action_dataset import MobileVLAActionDataset

def eval_int8_accuracy():
    # 1. Chunk5 Best Checkpoint 경로
    checkpoint_path = "/home/billy/25-1kp/vla/runs/mobile_vla_no_chunk_20251209/kosmos/mobile_vla_finetune/2025-12-17/mobile_vla_chunk5_20251217/epoch_epoch=06-val_loss=val_loss=0.067.ckpt"
    
    # 2. Config 설정 (INT8)
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0
    )
    
    print(f"🔄 Loading INT8 model from {Path(checkpoint_path).name}...")
    
    # 3. 모델 로드 (Trainer 없이 직접 모델 인스턴스화 또는 load_from_checkpoint)
    # MobileVLATrainer가 load_from_checkpoint 지원한다고 가정
    # quantization_config를 hparams나 init에 주입할 방법이 필요.
    # 보통 load_from_checkpoint는 **kwargs를 __init__으로 보냄.
    
    try:
        model = MobileVLATrainer.load_from_checkpoint(
            checkpoint_path,
            quantization_config=quantization_config,
            map_location="cuda"
        )
        print("✅ Model loaded successfully (INT8).")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # 4. Validation Dataloader 준비
    # 500개 데이터셋 기준 (20251204*)
    val_dataset = MobileVLAActionDataset(
        data_dir="/home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset",
        mode="val", # mode='val'이면 보통 전체 다 쓰거나 split 하는데, 여기선 일단 500개 패턴 지정이 중요
        # 패턴 지정 기능이 MobileVLAActionDataset에 없다면 상속받거나 외부에서 필터링해야 함.
        # MobileVLAActionDataset은 init에서 glob("*.h5")함.
        # 임시로 data_dir 내 파일을 필터링하는 로직이 필요.
    )
    
    # 데이터셋 필터링 (500개 맞추기: 20251204)
    # MobileVLAActionDataset.h5_files를 직접 수정
    filtered_files = [p for p in val_dataset.h5_files if "20251204" in p.name]
    val_dataset.h5_files = filtered_files
    print(f"📂 Dataset filtered: {len(val_dataset)} episodes (20251204 only)")
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=4, 
        shuffle=False, 
        num_workers=4,
        collate_fn=val_dataset.collate_fn if hasattr(val_dataset, 'collate_fn') else None
    )
    
    # 5. Validation Loop
    trainer = pl.Trainer(accelerator="gpu", devices=1, logger=False)
    
    print("🚀 Starting validation...")
    results = trainer.validate(model, dataloaders=val_loader)
    
    print("\n📊 INT8 Validation Results:")
    print(results)
    
    # 결과 저장
    import json
    with open("docs/int8_accuracy_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    eval_int8_accuracy()
