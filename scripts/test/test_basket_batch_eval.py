#!/usr/bin/env python3
"""
Basket Navigation 모델 대량 검증 스크립트
랜덤하게 10-20개의 에피소드를 선택하여 정확도(Perfect Match Rate) 측정
"""

import torch
import numpy as np
import h5py
import json
import random
from pathlib import Path
from PIL import Image
import sys
from tqdm import tqdm

# Add RoboVLMs to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "RoboVLMs_upstream"))
from robovlms.train.mobile_vla_trainer import MobileVLATrainer
from transformers import AutoProcessor

def snap_to_grid(val, is_y=False):
    """
    Y축 편향(Positive Bias)을 보정하고 전반적인 Gain을 높임
    """
    if is_y:
        # 모델이 수평 방향(Y)으로 항상 0.2~0.3 정도 양수 편향되어 있음
        # 이를 보정하기 위해 -0.25 시프트 후 2배 Gain 적용
        amplified_val = (val - 0.22) * 2.5
    else:
        # X축은 clipping 현상 보완을 위해 1.4배 Gain 적용
        amplified_val = val * 1.4
    
    movement_threshold = 0.35
    if amplified_val > movement_threshold:
        return 1.15
    elif amplified_val < -movement_threshold:
        return -1.15
    else:
        return 0.0

def run_batch_test(num_episodes=15):
    # 경로 설정
    checkpoint_path = "runs/mobile_vla_basket_chunk5/kosmos/mobile_vla_finetune/2026-01-29/mobile_vla_chunk5_basket_20260129/epoch_epoch=04-val_loss=val_loss=0.020.ckpt"
    config_path = "Mobile_VLA/configs/mobile_vla_chunk5_basket.json"
    dataset_dir = Path("/home/billy/25-1kp/vla/ROS_action/basket_dataset")
    
    # 모델 및 프로세서 로드
    print(f"🔧 모델 로드 중: {Path(checkpoint_path).name}")
    trainer = MobileVLATrainer.load_from_checkpoint(
        checkpoint_path,
        config_path=config_path,
        map_location="cuda"
    )
    trainer.model.to('cuda')
    trainer.model.eval()
    
    processor = AutoProcessor.from_pretrained('.vlms/kosmos-2-patch14-224')
    
    # 데이터셋 파일 리스트 확보
    all_files = list(dataset_dir.glob("*.h5"))
    if not all_files:
        print("❌ 데이터셋 파일을 찾을 수 없습니다.")
        return
        
    test_files = random.sample(all_files, min(num_episodes, len(all_files)))
    print(f"📂 {len(test_files)} 개의 에피소드 테스트 시작...\n")
    
    total_frames_tested = 0
    perfect_matches = 0
    total_rmse = 0
    all_errors = []
    all_raw_preds = []
    all_tested_files = []
    
    results = []

    for file_path in tqdm(test_files, desc="Processing Episodes"):
        episode_success = 0
        episode_frames = 0
        
        with h5py.File(file_path, 'r') as f:
            images = f['images'][:]
            actions = f['actions'][:]
            total_len = len(images)
            window_size = 8
            
            # 각 에피소드에서 3개의 랜덤 지점 테스트
            if total_len <= window_size:
                continue
                
            test_indices = random.sample(range(window_size-1, total_len), min(3, total_len - window_size + 1))
            
            for idx in test_indices:
                # Window input 준비
                start = idx - window_size + 1
                window_imgs = [Image.fromarray(images[i]) for i in range(start, idx + 1)]
                true_action = actions[idx][:2] # linear_x, linear_y
                
                # 지시어 결정 (Standard RoboVLMs format)
                instruction_raw = "Navigate to the brown pot on the left" if "left" in file_path.name else "Navigate to the brown pot on the right"
                instruction = f"<grounding>An image of a robot {instruction_raw}"
                
                # VLM 전처리
                all_pixel_values = []
                for img in window_imgs:
                    inputs = processor(images=img, text=instruction, return_tensors="pt")
                    all_pixel_values.append(inputs['pixel_values'])
                
                pixel_values = torch.cat(all_pixel_values, dim=0).unsqueeze(0).cuda()
                lang_x = inputs['input_ids'].cuda()
                attention_mask = inputs['attention_mask'].cuda()
                
                # 추론
                with torch.no_grad():
                    pred_output = trainer.model.inference(
                        pixel_values, lang_x, attention_mask,
                        None, None, None, None, None
                    )
                
                # 결과 추출
                action_out = pred_output['action']
                if isinstance(action_out, tuple): action_out = action_out[0]
                pred_raw = action_out[0, -1, 0].cpu().numpy()
                all_raw_preds.append(pred_raw)
                all_tested_files.append(file_path.name)
                
                # Snap 적용
                pred_snapped = np.array([snap_to_grid(pred_raw[0]), snap_to_grid(pred_raw[1], is_y=True)])
                
                # 통계 계산
                is_match = np.allclose(pred_snapped, true_action, atol=0.01)
                if is_match:
                    perfect_matches += 1
                    episode_success += 1
                
                error = pred_raw - true_action
                all_errors.append(error)
                
                rmse = np.sqrt(np.mean(error**2))
                total_rmse += rmse
                
                total_frames_tested += 1
                episode_frames += 1
        
        results.append({
            "file": file_path.name,
            "accuracy": (episode_success / episode_frames * 100) if episode_frames > 0 else 0
        })

    # 최종 결과 출력
    all_errors = np.array(all_errors)
    raw_vals = np.array(all_raw_preds)
    
    left_y = [raw_vals[i, 1] for i in range(len(raw_vals)) if "left" in all_tested_files[i]]
    right_y = [raw_vals[i, 1] for i in range(len(raw_vals)) if "right" in all_tested_files[i]]

    print("\n" + "="*70)
    print("📊 Basket Navigation 모델 방향성 분석")
    print("="*70)
    print(f"✅ 테스트 에피소드 수: {len(test_files)}")
    print(f"✅ 총 테스트 프레임: {total_frames_tested}")
    print(f"✅ Perfect Match Rate: {(perfect_matches/total_frames_tested*100):.2f}%")
    print("-" * 70)
    print(f"📈 [LEFT episodes] Raw Y Mean: {np.mean(left_y):.4f} (Should be Positive > 0)")
    print(f"📈 [RIGHT episodes] Raw Y Mean: {np.mean(right_y):.4f} (Should be Negative < 0)")
    print("-" * 70)
    print(f"📈 Raw Action Range: X=[{raw_vals[:,0].min():.2f}, {raw_vals[:,0].max():.2f}], Y=[{raw_vals[:,1].min():.2f}, {raw_vals[:,1].max():.2f}]")
    print("-" * 70)
    
    for res in results:
        print(f"- {res['file'][:50]:<50} | Success: {res['accuracy']:>6.1f}%")
    print("="*70)

if __name__ == "__main__":
    run_batch_test(15)
