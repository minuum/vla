import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from pytorch_lightning import Trainer

# 로컬 모듈 임포트
import sys
sys.path.append("/home/billy/25-1kp/vla/RoboVLMs_upstream")

from robovlms.train.mobile_vla_trainer import MobileVLATrainer
from robovlms.data.mobile_vla_action_dataset import MobileVLAActionDataset
from torch.utils.data import DataLoader

def check_model_scaling():
    # 1. Checkpoint Load
    ckpt_path = "/home/billy/25-1kp/vla/runs/mobile_vla_no_chunk_20251209/kosmos/mobile_vla_finetune/2025-12-17/mobile_vla_chunk5_20251217/epoch_epoch=06-val_loss=val_loss=0.067.ckpt"
    
    print(f"🔄 Loading model from {Path(ckpt_path).name}...")
    try:
        model = MobileVLATrainer.load_from_checkpoint(ckpt_path, map_location="cpu")
        model.eval()
        print("✅ Model loaded.")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # 2. Dataset Load (Validation set - 500 episodes)
    # 데이터셋 설정에서 norm_min/max 확인 (기본값 사용 시 -1.0, 1.0)
    val_dataset = MobileVLAActionDataset(
        data_dir="/home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset",
        mode="val",
        model_name="kosmos",
        norm_action=True,
        norm_min=-1.0,  # 학습 시 설정으로 추정
        norm_max=1.0,
    )
    # 202512 필터링
    val_dataset.h5_files = [f for f in val_dataset.h5_files if "202512" in f.name]
    print(f"📂 Validation Dataset: {len(val_dataset)} episodes")

    dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4, collate_fn=val_dataset.collate_fn)

    # 3. Inference & Collect Predictions
    all_preds = []
    all_gts = []
    
    print("🚀 Running inference...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 10: break # 10배치(80개)만 확인
            
            # Forward pass (MobileVLATrainer.predict_step 내부 로직 모사)
            # MobileVLATrainer는 forward에서 loss를 반환하므로, 모델 내부 구조를 직접 호출해야 함
            # model.model은 MobileVLAPolicy
            
            # 입력 준비
            # MobileVLA 모델 구조에 따라 다를 수 있음. 
            # 보통 model.predict_step 구현 여부 확인 없으면 직접 호출
            
            # 간단히 model.validation_step을 이용해 loss가 아닌 pred를 뽑을 수 있는지 확인하거나
            # model.model.generate() 또는 forward() 사용
            
            # 여기서는 model.model을 직접 사용한다고 가정
            pixel_values = batch['rgb']
            input_ids = batch['input_ids']
            
            # MobileVLAPolicy의 forward 리턴값 확인 필요하나, 
            # action head가 있는 경우 보통 action_logits를 반환
            
            # Trainer의 predict_step을 사용하는게 가장 안전
            pass
    
    # Trainer를 사용한 예측
    trainer = Trainer(accelerator="gpu", devices=1, logger=False)
    predictions = trainer.predict(model, dataloader)
    
    # predictions 구조: List[Dict] (MobileVLATrainer.predict_step 구현에 따름)
    # 만약 구현 안되어있으면 validation_step 루프를 직접 돌려야 함.
    
    # MobileVLATrainer 소스 확인이 어렵다면, 간단히 데이터셋의 raw값(GT) 분포만이라도 먼저 확인
    # Dataset이 반환하는 action은 이미 normalize된 값임.
    
    print("\n🔍 Checking Dataset Normalized Actions (Ground Truth):")
    gt_norm_actions = []
    for i in range(50): # 50개 샘플
        item = val_dataset[i]
        gt_norm_actions.append(item['action_tensors'])
    
    gt_norm_actions = np.concatenate(gt_norm_actions, axis=0) # [N, 7]
    
    print(f"GT Normalized Shape: {gt_norm_actions.shape}")
    print(f"GT Min: {gt_norm_actions.min(axis=0)}")
    print(f"GT Max: {gt_norm_actions.max(axis=0)}")
    
    # linear_x (idx 0), linear_y (idx 1)
    lin_x = gt_norm_actions[:, 0]
    lin_y = gt_norm_actions[:, 1]
    
    print(f"\nStats for linear_x (Normalized):")
    print(f"  Min: {lin_x.min():.4f}")
    print(f"  Max: {lin_x.max():.4f} (Should be <= 1.0)")
    
    # 클리핑 여부 확인
    clipping_ratio_x = np.sum(np.abs(lin_x) >= 0.999) / len(lin_x) * 100
    clipping_ratio_y = np.sum(np.abs(lin_y) >= 0.999) / len(lin_y) * 100
    
    print(f"  Clipping Ratio (Values near +/- 1.0): {clipping_ratio_x:.2f}%")
    
    print(f"\nStats for linear_y (Normalized):")
    print(f"  Min: {lin_y.min():.4f}")
    print(f"  Max: {lin_y.max():.4f}")
    print(f"  Clipping Ratio: {clipping_ratio_y:.2f}%")

    if clipping_ratio_x > 5.0 or clipping_ratio_y > 5.0:
        print("\n⚠️  Warning: Significant clipping detected! Data range exceeds norm_min/max.")
    else:
        print("\n✅ Data scaling looks reasonable (within range).")

if __name__ == "__main__":
    check_model_scaling()
