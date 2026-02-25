#!/usr/bin/env python3
import os
import sys
import json
import torch
import numpy as np
from tqdm import tqdm
from collections import Counter

sys.path.insert(0, os.path.abspath('RoboVLMs_upstream'))

from robovlms.train.mobile_vla_trainer import MobileVLATrainer
from robovlms.data.mobile_vla_h5_dataset import MobileVLAH5Dataset
from robovlms.utils.model_utils import build_tokenizer

# 패치
import robovlms.utils.model_utils as mode_utils
orig_dtc = mode_utils.default_tokenizer_config

def default_tokenizer_config_patch(tokenizer):
    if tokenizer == 'kosmos':
        return {'type': 'AutoProcessor', 'pretrained_model_name_or_path': 'microsoft/kosmos-2-patch14-224', 'tokenizer_type': 'kosmos'}
    try:
        return orig_dtc(tokenizer)
    except Exception:
        return {'type': 'AutoProcessor', 'pretrained_model_name_or_path': 'microsoft/kosmos-2-patch14-224', 'tokenizer_type': 'kosmos'}

mode_utils.default_tokenizer_config = default_tokenizer_config_patch

DATASET_DIR = "/home/billy/25-1kp/vla/ROS_action/basket_dataset_v2"
TOLERANCE = 0.01

def evaluate_pm_dm(checkpoint_path, config_path):
    print(f"Loading config from {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)
    print("Initializing trainer and loading model...")
    trainer = MobileVLATrainer.load_from_checkpoint(checkpoint_path, config_path=config_path, map_location="cuda")
    model = trainer.model.to('cuda')
    model.eval()

    tokenizer_config = config.get('tokenizer', None)
    tokenizer = build_tokenizer(tokenizer_config) if tokenizer_config else None
    dataset_tokenizer = tokenizer.tokenizer if hasattr(tokenizer, 'tokenizer') else tokenizer

    print("Loading test dataset...")
    val_dataset = MobileVLAH5Dataset(
        data_dir=DATASET_DIR,
        episode_pattern="episode_*.h5",
        model_name=config.get('val_dataset', {}).get('model_name', 'MobileVLA'),
        train_split=0.0,
        is_validation=True,
        window_size=config['window_size'],
        fwd_pred_next_n=config['fwd_pred_next_n'],
        tokenizer=dataset_tokenizer,
        tokenizer_config=tokenizer_config
    )

    num_eval = min(200, len(val_dataset))
    indices = np.linspace(0, len(val_dataset)-1, num_eval, dtype=int)

    pm_count = 0
    dm_count = 0

    print("Evaluating...")
    with torch.no_grad():
        for idx in tqdm(indices):
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

            pred_action = pred_logits.cpu().numpy()[0, -1, 0][:2] # X, Y
            gt_action = gpu_batch['action_chunck'].cpu().numpy()[0, -1, 0][:2]

            # Perfect Match
            if np.allclose(gt_action, pred_action, atol=TOLERANCE):
                pm_count += 1

            # Direction Match
            if (np.sign(gt_action[1]) == np.sign(pred_action[1])) or (abs(gt_action[1]) < 0.1 and abs(pred_action[1]) < 0.1):
                dm_count += 1
                
    pm_rate = (pm_count / num_eval) * 100
    dm_rate = (dm_count / num_eval) * 100
    
    print("\n" + "="*50)
    print(f"Results for Merged LoRA (v3_exp04)")
    print(f"Total Samples Tested : {num_eval}")
    print(f"PM (Perfect Match)   : {pm_rate:.2f}%")
    print(f"DM (Direction Match) : {dm_rate:.2f}%")
    print("="*50)

if __name__ == "__main__":
    ckpt = "/home/billy/25-1kp/vla/RoboVLMs_upstream/runs/v3_classification/kosmos/mobile_vla_v3_exp04_lora/2026-02-22/v3-exp04-lora/merged_v3_exp04_best.ckpt"
    cfg = "/home/billy/25-1kp/vla/Mobile_VLA/configs/mobile_vla_v3_exp04_inference.json"
    evaluate_pm_dm(ckpt, cfg)
