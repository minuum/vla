#!/usr/bin/env python3
import torch
import numpy as np
from pathlib import Path
import json
import sys
import os
from tqdm import tqdm

# Add RoboVLMs to path
sys.path.insert(0, os.path.abspath('RoboVLMs_upstream'))

from robovlms.train.mobile_vla_trainer import MobileVLATrainer
from robovlms.data.mobile_vla_h5_dataset import MobileVLAH5Dataset
from robovlms.utils.model_utils import build_tokenizer

def evaluate_accuracy(checkpoint_path, config_path, model_name):
    print(f"\nEvaluated Model: {model_name}")
    with open(config_path) as f:
        config = json.load(f)
    
    trainer = MobileVLATrainer.load_from_checkpoint(checkpoint_path, config_path=config_path, map_location="cuda")
    model = trainer.model.to('cuda')
    model.eval()
    
    tokenizer_config = config.get('tokenizer', None)
    tokenizer = build_tokenizer(tokenizer_config) if tokenizer_config else None
    dataset_tokenizer = tokenizer.tokenizer if hasattr(tokenizer, 'tokenizer') else tokenizer

    val_cfg = config['val_dataset']
    val_dataset = MobileVLAH5Dataset(
        data_dir=val_cfg['data_dir'],
        episode_pattern=val_cfg['episode_pattern'],
        model_name=val_cfg['model_name'],
        train_split=val_cfg['train_split'],
        is_validation=True,
        window_size=config['window_size'],
        fwd_pred_next_n=config['fwd_pred_next_n'],
        tokenizer=dataset_tokenizer,
        tokenizer_config=tokenizer_config
    )

    count = 0
    correct = 0
    total_mae_x = 0
    total_mae_y = 0
    
    num_eval = min(100, len(val_dataset)) # Increase to 100 for better statistics
    indices = np.linspace(0, len(val_dataset)-1, num_eval, dtype=int)
    
    with torch.no_grad():
        for idx in tqdm(indices, desc=f"Evaluating {model_name}"):
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
            
            pred_chunk = pred_logits.cpu().numpy()[0, -1]
            pred_action = pred_chunk[0]
            
            gt_chunk = gpu_batch['action_chunck'].cpu().numpy()[0, -1]
            gt_action = gt_chunk[0]
            
            # MAE
            total_mae_x += abs(pred_action[0] - gt_action[0])
            total_mae_y += abs(pred_action[1] - gt_action[1])
            
            # Direction Accuracy
            # gt_action[1] is linear_y. 1.0 (Left), -1.0 (Right), 0.0 (Straight)
            if np.sign(pred_action[1]) == np.sign(gt_action[1]):
                correct += 1
            elif abs(gt_action[1]) < 0.2 and abs(pred_action[1]) < 0.2: # Both straight enough
                correct += 1
                
            count += 1

    return {
        'mae_x': total_mae_x / count,
        'mae_y': total_mae_y / count,
        'accuracy': correct / count
    }

if __name__ == "__main__":
    v2_17_ckpt = "/home/billy/25-1kp/vla/RoboVLMs_upstream/runs/exp_v2_series/kosmos/mobile_vla_exp_v2_17/2026-02-15/exp-v2-17/epoch_epoch=09-val_loss=val_loss=0.001.ckpt"
    v2_17_config = "/home/billy/25-1kp/vla/Mobile_VLA/configs/mobile_vla_exp_v2_17.json"
    
    v2_12_ckpt = "/home/billy/25-1kp/vla/RoboVLMs_upstream/runs/exp_v2_series/kosmos/mobile_vla_exp_v2_12/2026-02-16/exp-v2-12/epoch_epoch=07-val_loss=val_loss=0.001.ckpt"
    v2_12_config = "/home/billy/25-1kp/vla/Mobile_VLA/configs/mobile_vla_exp_v2_12.json"

    res17 = evaluate_accuracy(v2_17_ckpt, v2_17_config, "EXP-V2-17")
    res12 = evaluate_accuracy(v2_12_ckpt, v2_12_config, "EXP-V2-12")
    
    print("\n" + "="*60)
    print(f"{'Experiment':<15} | {'MAE X':<10} | {'MAE Y':<10} | {'Accuracy':<10}")
    print("-" * 60)
    print(f"{'EXP-V2-17':<15} | {res17['mae_x']:<10.4f} | {res17['mae_y']:<10.4f} | {res17['accuracy']*100:<9.1f}%")
    print(f"{'EXP-V2-12':<15} | {res12['mae_x']:<10.4f} | {res12['mae_y']:<10.4f} | {res12['accuracy']*100:<9.1f}%")
    print("="*60)
