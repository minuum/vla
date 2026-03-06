#!/usr/bin/env python3
import os
import sys
import json
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import Counter

# Inject Custom Code Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "third_party" / "RoboVLMs"))

import robovlms.data
import robovlms.model.policy_head
import robovlms.model.backbone
import robovlms.train

# Custom Components
from robovlm_nav.datasets.nav_dataset import NavDataset
from robovlm_nav.datasets.nav_h5_dataset_impl import MobileVLAH5Dataset as NavH5DatasetImpl
from robovlm_nav.models.policy_head.nav_policy_impl import (
    MobileVLALSTMDecoder as NavLSTMDecoder,
    MobileVLAClassificationDecoder as NavClassificationDecoder,
)
from robovlm_nav.trainer.nav_trainer import MobileVLATrainer as NavTrainer

# Dynamic Injection into RoboVLMs namespace
setattr(robovlms.data, "NavDataset", NavDataset)
setattr(robovlms.data, "MobileVLAH5Dataset", NavH5DatasetImpl)
setattr(robovlms.model.policy_head, "NavPolicy", NavClassificationDecoder)
setattr(robovlms.model.policy_head, "NavPolicyRegression", NavLSTMDecoder)
setattr(robovlms.model.policy_head, "MobileVLAClassificationDecoder", NavClassificationDecoder)
setattr(robovlms.model.policy_head, "MobileVLALSTMDecoder", NavLSTMDecoder)

from robovlms.model.backbone.robokosmos import RoboKosMos
setattr(robovlms.model.backbone, "RoboVLM-Nav", RoboKosMos)
setattr(robovlms.train, "MobileVLATrainer", NavTrainer)

from robovlms.utils.model_utils import build_tokenizer
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

# Mapping
discrete_map = [
    [0.0, 0.0],    # 0: STOP
    [0.3, 0.0],    # 1: F
    [-0.3, 0.0],   # 2: B
    [0.0, 0.4],    # 3: L
    [0.0, -0.4],   # 4: R
    [0.3, 0.4],    # 5: FL
    [0.3, -0.4],   # 6: FR
    [-0.3, 0.4],   # 7: BL
    [-0.3, -0.4]   # 8: BR
]
TOLERANCE = 0.01

def evaluate_pm_dm(checkpoint_path, config_path):
    print(f"Loading config from {config_path}")
    with open(config_path, 'r') as f: config = json.load(f)
    print("Initializing trainer...")
    trainer = NavTrainer.load_from_checkpoint(checkpoint_path, config_path=config_path, map_location="cuda")
    model = trainer.model.to('cuda')
    model.eval()

    tokenizer_config = config.get('tokenizer', None)
    tokenizer = build_tokenizer(tokenizer_config) if tokenizer_config else None
    dataset_tokenizer = tokenizer.tokenizer if hasattr(tokenizer, 'tokenizer') else tokenizer

    print("Loading test dataset...")
    val_dataset = NavH5DatasetImpl(
        data_dir=config.get('val_dataset', {}).get('data_dir', '/home/billy/25-1kp/vla/ROS_action/basket_dataset_v2'),
        episode_pattern="episode_*.h5",
        model_name=config.get('val_dataset', {}).get('model_name', 'kosmos'),
        train_split=0.9, # ONLY use the last 10% for validation (unseen during training)
        is_validation=True,
        window_size=config['act_head']['window_size'],
        fwd_pred_next_n=config['act_head']['fwd_pred_next_n'],
        tokenizer=dataset_tokenizer,
        tokenizer_config=tokenizer_config,
        discrete_action=True,
        instruction_preset=config.get('val_dataset', {}).get('instruction_preset', 'center_goal')
    )

    num_eval = min(200, len(val_dataset))
    indices = np.linspace(0, len(val_dataset)-1, num_eval, dtype=int)
    
    pm_count = 0
    dm_count = 0
    
    dir_stats = {
        'Straight': {'total': 0, 'pm': 0, 'dm': 0},
        'Left': {'total': 0, 'pm': 0, 'dm': 0},
        'Right': {'total': 0, 'pm': 0, 'dm': 0},
        'Stop': {'total': 0, 'pm': 0, 'dm': 0}
    }
    
    print(f"Evaluating {num_eval} samples...")
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
            
            # Action logic: (B, Seq, Chunk, NumClasses)
            pred_class = pred_logits[0, -1, 0].argmax(dim=-1).item()
            pred_action = np.array(discrete_map[pred_class])
            
            # GT action mapping
            # action_chunck has shape [1, Seq, Chunk] and contains class indices when discrete_action is True
            gt_class = gpu_batch['action_chunck'].cpu().numpy()[0, -1, 0]
            gt_action = np.array(discrete_map[int(gt_class)])

            # Direction category
            c = int(gt_class)
            if c == 0: cat = 'Stop'
            elif c in [1, 2]: cat = 'Straight'
            elif c in [3, 5, 7]: cat = 'Left'
            elif c in [4, 6, 8]: cat = 'Right'
            
            dir_stats[cat]['total'] += 1

            is_pm = False
            is_dm = False

            # GT mapped to class to compare PM
            # Let's just compare pred_action with gt_action
            if np.allclose(gt_action, pred_action, atol=TOLERANCE):
                pm_count += 1
                dir_stats[cat]['pm'] += 1
                is_pm = True
            elif (abs(gt_action[0]) < 0.1 and abs(pred_action[0]) < 0.1) and (abs(gt_action[1]) < 0.1 and abs(pred_action[1]) < 0.1): 
                # both STOP
                pm_count += 1
                dir_stats[cat]['pm'] += 1
                is_pm = True
                
            # Direction Match (Y-axis direction)
            if (np.sign(gt_action[1]) == np.sign(pred_action[1])) or (abs(gt_action[1]) < 0.1 and abs(pred_action[1]) < 0.1):
                dm_count += 1
                dir_stats[cat]['dm'] += 1
                is_dm = True
                
    print("\n" + "="*50)
    print(f"Results for Checkpoint: {os.path.basename(checkpoint_path)}")
    print(f"Total Samples Tested : {num_eval}")
    print(f"Overall PM (Perfect Match)   : {pm_count/num_eval*100:.2f}%")
    print(f"Overall DM (Direction Match) : {dm_count/num_eval*100:.2f}%")
    print("-" * 50)
    for k, v in dir_stats.items():
        if v['total'] > 0:
            print(f"{k} [{v['total']} samples] - PM: {v['pm']/v['total']*100:.2f}%, DM: {v['dm']/v['total']*100:.2f}%")
        else:
            print(f"{k} [0 samples] - N/A")
    print("="*50)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--cfg', type=str, required=True)
    args = parser.parse_args()
    evaluate_pm_dm(args.ckpt, args.cfg)
