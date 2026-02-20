#!/usr/bin/env python3
"""
V2 모델 상세 분석 스크립트 (Offline)
- 대상: EXP-V2-17, EXP-V2-12
- 기능: 구간별(초기/중기/후기) PM/DM 분석, 오류 유형 분류
"""

import os
import sys
import json
import torch
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict, Counter

# Add RoboVLMs to path
sys.path.insert(0, os.path.abspath('RoboVLMs_upstream'))

from robovlms.train.mobile_vla_trainer import MobileVLATrainer
from robovlms.data.mobile_vla_h5_dataset import MobileVLAH5Dataset
from robovlms.utils.model_utils import build_tokenizer

# 분석 설정
DATASET_DIR = "/home/billy/25-1kp/vla/ROS_action/basket_dataset_v2/test"
NUM_EPISODES = 20
TOLERANCE = 0.01  # Perfect Match 기준
STOP_THRESHOLD = 0.05

class ErrorClassifier:
    """오류 유형 분류기"""
    @staticmethod
    def classify_error(gt, pred):
        gt = np.array(gt).flatten()[:2]
        pred = np.array(pred).flatten()[:2]
        
        if np.allclose(gt, pred, atol=TOLERANCE):
            return 'perfect'
        
        gt_stopped = np.allclose(gt, [0, 0], atol=STOP_THRESHOLD)
        pred_stopped = np.allclose(pred, [0, 0], atol=STOP_THRESHOLD)
        
        if gt_stopped and not pred_stopped:
            return 'stop_confusion_false_move'
        if not gt_stopped and pred_stopped:
            return 'stop_confusion_false_stop'
        
        gt_x, gt_y = gt
        pred_x, pred_y = pred
        if (gt_x * pred_x < -1e-3) or (gt_y * pred_y < -1e-3):
            return 'direction_flip'
            
        mag_gt = np.linalg.norm(gt)
        mag_pred = np.linalg.norm(pred)
        if mag_gt > 1e-3:
            if mag_pred < mag_gt * 0.7: return 'magnitude_under'
            if mag_pred > mag_gt * 1.3: return 'magnitude_over'
        
        return 'minor_deviation'

def evaluate_model_detailed(checkpoint_path, config_path, model_label):
    print(f"\n[Testing Model: {model_label}]")
    print(f"Loading checkpoint: {checkpoint_path}")
    
    with open(config_path) as f:
        config = json.load(f)
    
    trainer = MobileVLATrainer.load_from_checkpoint(checkpoint_path, config_path=config_path, map_location="cuda")
    model = trainer.model.to('cuda')
    model.eval()
    
    tokenizer_config = config.get('tokenizer', None)
    tokenizer = build_tokenizer(tokenizer_config) if tokenizer_config else None
    dataset_tokenizer = tokenizer.tokenizer if hasattr(tokenizer, 'tokenizer') else tokenizer

    val_dataset = MobileVLAH5Dataset(
        data_dir=DATASET_DIR,
        episode_pattern="episode_*.h5",
        model_name=config['val_dataset']['model_name'],
        train_split=0.0, # Test set only
        is_validation=True,
        window_size=config['window_size'],
        fwd_pred_next_n=config['fwd_pred_next_n'],
        tokenizer=dataset_tokenizer,
        tokenizer_config=tokenizer_config
    )

    all_results = []
    aggregated_errors = Counter()
    
    # 에피소드별로 묶어서 분석하기 위해 인덱스 계산 (단순 샘플링 대신 에피소드 단위 분석 지향)
    # 하지만 Dataset 구조상 슬라이딩 윈도우이므로, 여기서는 주요 샘플 100개를 분석
    num_eval = min(100, len(val_dataset))
    indices = np.linspace(0, len(val_dataset)-1, num_eval, dtype=int)
    
    frame_details = []

    with torch.no_grad():
        for idx in tqdm(indices, desc=f"Analyzing {model_label}"):
            sample = val_dataset[idx]
            batch = val_dataset.collater([sample])
            gpu_batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            prediction = model.inference(
                gpu_batch['rgb'],
                gpu_batch['text'],
                attention_mask=gpu_batch['text_mask'],
                vision_gripper=gpu_batch['hand_rgb'],
                raw_text=gpu_batch['raw_text'],
                data_source=gpu_batch['data_source']
            )
            
            pred_logits = prediction['action']
            if isinstance(pred_logits, tuple): pred_logits = pred_logits[0]
            
            # Continuous action extraction
            pred_action = pred_logits.cpu().numpy()[0, -1, 0] # (B, seq, chunk, dim) -> last step, first chunk
            gt_action = gpu_batch['action_chunck'].cpu().numpy()[0, -1, 0]
            
            # Logic for phase determination (approximate based on idx)
            # In a real episode, we'd use metadata, here we'll use a simple heuristic for demonstration
            # or just focus on Global for now if metadata is missing.
            
            error_type = ErrorClassifier.classify_error(gt_action, pred_action)
            perfect = (error_type == 'perfect')
            dir_match = (np.sign(gt_action[1]) == np.sign(pred_action[1])) or (abs(gt_action[1]) < 0.1 and abs(pred_action[1]) < 0.1)
            
            frame_info = {
                'perfect': perfect,
                'dir_match': dir_match,
                'error_type': error_type,
                'error_dist': np.linalg.norm(gt_action - pred_action)
            }
            frame_details.append(frame_info)
            aggregated_errors[error_type] += 1

    pm_rate = sum(1 for f in frame_details if f['perfect']) / len(frame_details) * 100
    dm_rate = sum(1 for f in frame_details if f['dir_match']) / len(frame_details) * 100
    
    print(f"\n--- {model_label} Results ---")
    print(f"Global Perfect Match: {pm_rate:.2f}%")
    print(f"Global Direction Agreement: {dm_rate:.2f}%")
    print("\nError Distribution:")
    for et, count in aggregated_errors.most_common():
        print(f"  {et:25s}: {count:3d} ({count/len(frame_details)*100:5.1f}%)")
    
    return {
        'model': model_label,
        'pm': pm_rate,
        'dm': dm_rate,
        'errors': dict(aggregated_errors)
    }

if __name__ == "__main__":
    v2_17_ckpt = "/home/billy/25-1kp/vla/RoboVLMs_upstream/runs/exp_v2_series/kosmos/mobile_vla_exp_v2_17/2026-02-15/exp-v2-17/epoch_epoch=09-val_loss=val_loss=0.001.ckpt"
    v2_17_config = "/home/billy/25-1kp/vla/Mobile_VLA/configs/mobile_vla_exp_v2_17.json"
    
    v2_12_ckpt = "/home/billy/25-1kp/vla/RoboVLMs_upstream/runs/exp_v2_series/kosmos/mobile_vla_exp_v2_12/2026-02-16/exp-v2-12/epoch_epoch=07-val_loss=val_loss=0.001.ckpt"
    v2_12_config = "/home/billy/25-1kp/vla/Mobile_VLA/configs/mobile_vla_exp_v2_12.json"

    results = []
    if os.path.exists(v2_17_ckpt):
        results.append(evaluate_model_detailed(v2_17_ckpt, v2_17_config, "EXP-V2-17"))
    else:
        print(f"Error: {v2_17_ckpt} not found")
        
    if os.path.exists(v2_12_ckpt):
        results.append(evaluate_model_detailed(v2_12_ckpt, v2_12_config, "EXP-V2-12"))
    else:
        print(f"Error: {v2_12_ckpt} not found")

    print("\n" + "="*80)
    print(f"{'Model':<15} | {'PM Rate':<12} | {'DM Rate':<12} | {'Top Error':<20}")
    print("-" * 80)
    for r in results:
        top_err = sorted(r['errors'].items(), key=lambda x: x[1], reverse=True)
        top_err_str = top_err[1][0] if len(top_err) > 1 else "None"
        print(f"{r['model']:<15} | {r['pm']:>10.2f}% | {r['dm']:>10.2f}% | {top_err_str:<20}")
    print("="*80)
