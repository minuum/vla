import torch
import json
import os
import sys

# Add project root to path
project_root = "/home/billy/25-1kp/vla"
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'RoboVLMs_upstream'))

from robovlms.train.mobile_vla_trainer import MobileVLATrainer

def check_keys():
    config_path = "/home/billy/25-1kp/vla/Mobile_VLA/configs/mobile_vla_exp_v2_17.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Initialize model
    model = MobileVLATrainer(config)
    model_keys = set(model.state_dict().keys())
    
    # Load checkpoint
    checkpoint_path = "runs/unified_regression_win12/kosmos/mobile_vla_exp17_win8_k1/2026-02-10/exp17_win8_k1/epoch=epoch=09-val_loss=val_loss=0.0013.ckpt"
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    ckpt_state = checkpoint.get('state_dict', checkpoint.get('model_state_dict'))
    ckpt_keys = set(ckpt_state.keys())
    
    print(f"Model Keys (Total): {len(model_keys)}")
    print(f"Ckpt Keys (Total): {len(ckpt_keys)}")
    
    # Check intersection
    match = model_keys.intersection(ckpt_keys)
    print(f"Direct Match: {len(match)}")
    
    # Check with 'model.' prefix removal
    stripped_ckpt_keys = {k.replace('model.', '', 1) for k in ckpt_keys}
    match_stripped = model_keys.intersection(stripped_ckpt_keys)
    print(f"Stripped Match: {len(match_stripped)}")
    
    # Check policy head specifically
    policy_keys = [k for k in model_keys if 'act_head' in k or 'policy_head' in k]
    print(f"\nPolicy Head Keys in Model: {len(policy_keys)}")
    
    policy_match = [k for k in policy_keys if k in ckpt_keys]
    print(f"Direct Policy Match: {len(policy_match)}")
    
    policy_match_stripped = [k for k in policy_keys if f"model.{k}" in ckpt_keys]
    print(f"Prefix Policy Match: {len(policy_match_stripped)}")

if __name__ == "__main__":
    check_keys()
