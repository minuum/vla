#!/usr/bin/env python3
"""
Velocity 출력 검증 스크립트
목적: 예측된 velocity가 합리적인지 검증
"""

import torch
import h5py
import numpy as np
from pathlib import Path
import argparse
import sys
import glob

# RoboVLMs 경로 추가
sys.path.insert(0, str(Path(__file__).parent / "RoboVLMs_upstream"))

from robovlms.train.mobile_vla_trainer import MobileVLATrainer
from PIL import Image
import torchvision.transforms as T


def verify_velocity_output(checkpoint_path, test_data_dir, num_samples=20):
    """
    Velocity 출력 검증
    
    Args:
        checkpoint_path: 체크포인트 경로
        test_data_dir: 테스트 데이터 디렉토리
        num_samples: 테스트 샘플 수
    """
    print("="*60)
    print("Velocity 출력 검증")
    print("="*60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Test Data: {test_data_dir}")
    print(f"Samples: {num_samples}")
    print()
    
    # 1. 모델 로드
    print("[1/5] 모델 로딩...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        model = MobileVLATrainer.load_from_checkpoint(checkpoint_path)
        model = model.to(device)
        model.eval()
        print(f"  ✅ 모델 로드 완료 (Device: {device})")
    except Exception as e:
        print(f"  ❌ 모델 로드 실패: {e}")
        return
    
    # 2. 테스트 데이터 로드
    print("\n[2/5] 테스트 데이터 로딩...")
    h5_files = sorted(glob.glob(f"{test_data_dir}/episode_*.h5"))[:num_samples]
    
    if len(h5_files) == 0:
        print(f"  ❌ 데이터 없음: {test_data_dir}")
        return
    
    print(f"  ✅ {len(h5_files)} episodes 로드")
    
    # 3. 예측 수행
    print("\n[3/5] Velocity 예측...")
    
    predicted_velocities = []
    ground_truth_velocities = []
    
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])
    
    with torch.no_grad():
        for i, h5_file in enumerate(h5_files):
            with h5py.File(h5_file, 'r') as f:
                # 이미지 로드 (첫 8프레임)
                images = []
                for t in range(min(8, len(f['images']))):
                    img_array = f['images'][t]
                    img = Image.fromarray(img_array.astype(np.uint8))
                    img_tensor = transform(img)
                    images.append(img_tensor)
                
                # Padding if needed
                while len(images) < 8:
                    images.append(torch.zeros(3, 224, 224))
                
                images_tensor = torch.stack(images).unsqueeze(0).to(device)  # (1, 8, 3, 224, 224)
                
                # 예측
                context = model.model.encode_images(images_tensor)
                actions = model.model.act_head(context)  # (1, 10, 2)
                
                pred_vel = actions[0, 0, :].cpu().numpy()  # 첫 timestep만
                predicted_velocities.append(pred_vel)
                
                # Ground truth
                gt_actions = f['actions'][min(8, len(f['actions'])-1)][:2]
                ground_truth_velocities.append(gt_actions)
            
            if (i + 1) % 5 == 0:
                print(f"  Progress: {i+1}/{len(h5_files)}")
    
    predicted_velocities = np.array(predicted_velocities)
    ground_truth_velocities = np.array(ground_truth_velocities)
    
    # 4. 통계 분석
    print("\n[4/5] 통계 분석...")
    
    print("\n📊 Predicted Velocity:")
    print(f"  Shape: {predicted_velocities.shape}")
    print(f"  linear_x: Mean={np.mean(predicted_velocities[:, 0]):.4f}, Std={np.std(predicted_velocities[:, 0]):.4f}")
    print(f"  linear_y: Mean={np.mean(predicted_velocities[:, 1]):.4f}, Std={np.std(predicted_velocities[:, 1]):.4f}")
    print(f"  Range: [{np.min(predicted_velocities):.4f}, {np.max(predicted_velocities):.4f}]")
    
    print("\n📊 Ground Truth Velocity:")
    print(f"  Shape: {ground_truth_velocities.shape}")
    print(f"  linear_x: Mean={np.mean(ground_truth_velocities[:, 0]):.4f}, Std={np.std(ground_truth_velocities[:, 0]):.4f}")
    print(f"  linear_y: Mean={np.mean(ground_truth_velocities[:, 1]):.4f}, Std={np.std(ground_truth_velocities[:, 1]):.4f}")
    print(f"  Range: [{np.min(ground_truth_velocities):.4f}, {np.max(ground_truth_velocities):.4f}]")
    
    # 5. 오차 분석
    print("\n[5/5] 오차 분석...")
    
    errors = predicted_velocities - ground_truth_velocities
    rmse = np.sqrt(np.mean(errors ** 2, axis=0))
    mae = np.mean(np.abs(errors), axis=0)
    
    print("\n📊 Error Metrics:")
    print(f"  RMSE: linear_x={rmse[0]:.4f}, linear_y={rmse[1]:.4f}")
    print(f"  MAE:  linear_x={mae[0]:.4f}, linear_y={mae[1]:.4f}")
    print(f"  Overall RMSE: {np.sqrt(np.mean(errors ** 2)):.4f}")
    
    # 6. 샘플 출력
    print("\n📋 Sample Predictions (첫 5개):")
    print("  Index | Pred linear_x | Pred linear_y | GT linear_x | GT linear_y | Error")
    print("  " + "-"*75)
    for i in range(min(5, len(predicted_velocities))):
        pred_x, pred_y = predicted_velocities[i]
        gt_x, gt_y = ground_truth_velocities[i]
        err = np.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)
        print(f"  {i:5d} | {pred_x:13.4f} | {pred_y:13.4f} | {gt_x:11.4f} | {gt_y:11.4f} | {err:.4f}")
    
    # 7. 합리성 검증
    print("\n" + "="*60)
    print("합리성 검증")
    print("="*60)
    
    # Check 1: 범위
    in_range = np.all((predicted_velocities >= -1.0) & (predicted_velocities <= 1.0))
    print(f"\n1. 범위 검증 ([-1, 1]):")
    if in_range:
        print(f"  ✅ 모든 값이 정규화 범위 내에 있습니다")
    else:
        print(f"  ⚠️  범위 밖 값 존재")
    
    # Check 2: RMSE
    target_rmse = 0.12
    overall_rmse = np.sqrt(np.mean(errors ** 2))
    print(f"\n2. RMSE 검증 (목표 < {target_rmse}):")
    if overall_rmse < target_rmse:
        print(f"  ✅ RMSE {overall_rmse:.4f} < {target_rmse}")
    else:
        print(f"  ⚠️  RMSE {overall_rmse:.4f} >= {target_rmse}")
    
    # Check 3: 분포
    print(f"\n3. 분포 검증:")
    print(f"  Predicted std: {np.std(predicted_velocities):.4f}")
    print(f"  GT std: {np.std(ground_truth_velocities):.4f}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Velocity 출력 검증")
    parser.add_argument("--checkpoint", type=str,
                       default="RoboVLMs_upstream/runs/mobile_vla_lora_20251203/kosmos/mobile_vla_finetune/2025-12-03/mobile_vla_lora_20251203/epoch_epoch=09-val_loss=val_loss=0.013.ckpt",
                       help="Checkpoint 경로")
    parser.add_argument("--test_data", type=str,
                       default="ROS_action/mobile_vla_dataset",
                       help="테스트 데이터 디렉토리")
    parser.add_argument("--samples", type=int, default=20,
                       help="테스트 샘플 수")
    
    args = parser.parse_args()
    
    verify_velocity_output(args.checkpoint, args.test_data, args.samples)
