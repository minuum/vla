import torch
import json
import os
import sys

# Add project root to path
project_root = "/home/billy/25-1kp/vla"
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'RoboVLMs_upstream'))

from robovlms.train.mobile_vla_trainer import MobileVLATrainer
from transformers import AutoProcessor

def check_sync():
    config_path = "/home/billy/25-1kp/vla/Mobile_VLA/configs/mobile_vla_exp_v2_17.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"--- Config Check ({config_path}) ---")
    print(f"norm_action: {config.get('norm_action')}")
    print(f"image_mean: {config.get('image_mean')}")
    print(f"image_std: {config.get('image_std')}")
    
    print("\n--- Processor Check ---")
    processor = AutoProcessor.from_pretrained(config['tokenizer']['pretrained_model_name_or_path'])
    if hasattr(processor, 'image_processor'):
        print(f"Processor Image Mean: {processor.image_processor.image_mean}")
        print(f"Processor Image Std: {processor.image_processor.image_std}")
    
    print("\n--- Prompt Check (KOSMOS) ---")
    from robovlms.data.data_utils import get_text_function
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer']['pretrained_model_name_or_path'])
    text_fn = get_text_function(tokenizer, "kosmos")
    input_ids, mask = text_fn(["test instruction"])
    decoded = tokenizer.decode(input_ids[0])
    print(f"Decoded Prompt Flow: '{decoded}'")
    
    print("\n--- Checkpoint Check ---")
    checkpoint_path = "runs/unified_regression_win12/kosmos/mobile_vla_exp17_win8_k1/2026-02-10/exp17_win8_k1/epoch=epoch=09-val_loss=val_loss=0.0013.ckpt"

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint.get('model_state_dict'))
    print(f"Checkpoint keys example: {list(state_dict.keys())[:5]}")
    
    # Check if 'model.' prefix exists
    has_model_prefix = any(k.startswith('model.') for k in state_dict.keys())
    print(f"Has 'model.' prefix: {has_model_prefix}")

if __name__ == "__main__":
    check_sync()
