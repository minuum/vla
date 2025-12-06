#!/usr/bin/env python3
"""
Left vs Right Action 구분 분석
목적: Action head가 left/right 방향을 구분하는지 확인
"""

import torch
import numpy as np
import h5py
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, "RoboVLMs_upstream")

from robovlms.train.mobile_vla_trainer import MobileVLATrainer


def load_model(checkpoint_path, device='cuda'):
    """모델 로드"""
    print(f"모델 로드: {Path(checkpoint_path).name}")
    model = MobileVLATrainer.load_from_checkpoint(checkpoint_path, map_location='cpu')
    model.eval()
    model.to(device)
    return model


def analyze_left_right_actions(model, device='cuda'):
    """Left vs Right action 분석"""
    print("\n" + "="*70)
    print("Left vs Right Action 분석")
    print("="*70)
    
    h5_files = sorted(list(Path("ROS_action/mobile_vla_dataset").glob("episode_*.h5")))
    
    # Left/Right 분리
    left_files = [f for f in h5_files if 'left' in str(f)][:25]
    right_files = [f for f in h5_files if 'right' in str(f)][:25]
    
    print(f"Left samples: {len(left_files)}")
    print(f"Right samples: {len(right_files)}")
    
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])
    
    left_predictions = []
    right_predictions = []
    left_gt = []
    right_gt = []
    
    with torch.no_grad():
        # Left samples
        print("\nProcessing Left samples...")
        for h5_file in left_files[:10]:  # 처음 10개만
            pred, gt = predict_single(model, h5_file, transform, device)
            if pred is not None:
                left_predictions.append(pred)
                left_gt.append(gt)
        
        # Right samples
        print("Processing Right samples...")
        for h5_file in right_files[:10]:  # 처음 10개만
            pred, gt = predict_single(model, h5_file, transform, device)
            if pred is not None:
                right_predictions.append(pred)
                right_gt.append(gt)
    
    left_predictions = np.array(left_predictions)
    right_predictions = np.array(right_predictions)
    left_gt = np.array(left_gt)
    right_gt = np.array(right_gt)
    
    # 분석
    print("\n" + "="*70)
    print("📊 결과 분석")
    print("="*70)
    
    print("\n=== Predicted Velocities ===")
    print("\n[LEFT samples] (should go left → negative linear_y expected)")
    print(f"  linear_x: mean={left_predictions[:, 0].mean():.4f}, std={left_predictions[:, 0].std():.4f}")
    print(f"  linear_y: mean={left_predictions[:, 1].mean():.4f}, std={left_predictions[:, 1].std():.4f}")
    
    print("\n[RIGHT samples] (should go right → positive linear_y expected)")
    print(f"  linear_x: mean={right_predictions[:, 0].mean():.4f}, std={right_predictions[:, 0].std():.4f}")
    print(f"  linear_y: mean={right_predictions[:, 1].mean():.4f}, std={right_predictions[:, 1].std():.4f}")
    
    print("\n=== Ground Truth Velocities ===")
    print("\n[LEFT samples]")
    print(f"  linear_x: mean={left_gt[:, 0].mean():.4f}, std={left_gt[:, 0].std():.4f}")
    print(f"  linear_y: mean={left_gt[:, 1].mean():.4f}, std={left_gt[:, 1].std():.4f}")
    
    print("\n[RIGHT samples]")
    print(f"  linear_x: mean={right_gt[:, 0].mean():.4f}, std={right_gt[:, 0].std():.4f}")
    print(f"  linear_y: mean={right_gt[:, 1].mean():.4f}, std={right_gt[:, 1].std():.4f}")
    
    # 방향 구분 분석
    print("\n" + "="*70)
    print("🔍 방향 구분 분석")
    print("="*70)
    
    left_y_mean = left_predictions[:, 1].mean()
    right_y_mean = right_predictions[:, 1].mean()
    diff = right_y_mean - left_y_mean
    
    print(f"\nPredicted linear_y 차이: {diff:.4f}")
    print(f"  Left mean: {left_y_mean:.4f}")
    print(f"  Right mean: {right_y_mean:.4f}")
    
    if diff > 0.1:
        print("\n✅ Action head가 Left/Right를 구분하고 있음!")
        print("   Right가 더 positive linear_y (올바름)")
    elif diff < -0.1:
        print("\n⚠️ Action head가 반대로 학습됨!")
        print("   Left가 더 positive linear_y")
    else:
        print("\n❌ Action head가 Left/Right를 구분하지 못함")
        print("   차이가 너무 작음 (< 0.1)")
    
    # Sample-wise 비교
    print("\n=== Sample-wise 비교 ===")
    print("\nLeft samples:")
    for i, (pred, gt) in enumerate(zip(left_predictions[:5], left_gt[:5])):
        print(f"  {i}: pred=({pred[0]:.3f}, {pred[1]:.3f}), gt=({gt[0]:.3f}, {gt[1]:.3f})")
    
    print("\nRight samples:")
    for i, (pred, gt) in enumerate(zip(right_predictions[:5], right_gt[:5])):
        print(f"  {i}: pred=({pred[0]:.3f}, {pred[1]:.3f}), gt=({gt[0]:.3f}, {gt[1]:.3f})")
    
    return {
        'left_pred': left_predictions,
        'right_pred': right_predictions,
        'left_gt': left_gt,
        'right_gt': right_gt
    }


def predict_single(model, h5_file, transform, device):
    """단일 샘플 예측"""
    try:
        with h5py.File(h5_file, 'r') as f:
            # 이미지 로드
            images = []
            for t in range(min(8, len(f['images']))):
                img = Image.fromarray(f['images'][t].astype(np.uint8))
                images.append(transform(img))
            
            while len(images) < 8:
                images.append(torch.zeros(3, 224, 224))
            
            images_tensor = torch.stack(images).unsqueeze(0).to(device)
            
            # 예측
            context = model.model.encode_images(images_tensor)
            batch_size = context.shape[0]
            context_flat = context.view(batch_size, -1, context.shape[-1])
            action_mask = torch.ones(batch_size, 8, dtype=torch.bool).to(device)
            
            actions = model.model.act_head(context_flat, actions=None, action_masks=action_mask)
            
            if isinstance(actions, tuple):
                actions = actions[0]
            
            # 첫 번째 예측 (첫 토큰, 첫 timestep)
            pred = actions[0, 0, 0, :2].cpu().numpy()
            
            # Ground truth (8번째 프레임의 action)
            gt = f['actions'][min(8, len(f['actions'])-1)][:2]
            
            return pred, gt
    except Exception as e:
        print(f"  Error with {h5_file.name}: {e}")
        return None, None


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    checkpoint = "RoboVLMs_upstream/runs/mobile_vla_kosmos2_frozen_lora_leftright_20251204/kosmos/mobile_vla_finetune/2025-12-04/mobile_vla_kosmos2_frozen_lora_leftright_20251204/epoch_epoch=08-val_loss=val_loss=0.027.ckpt"
    
    model = load_model(checkpoint, device)
    results = analyze_left_right_actions(model, device)
    
    print("\n" + "="*70)
    print("✅ 분석 완료")
    print("="*70)


if __name__ == "__main__":
    main()
