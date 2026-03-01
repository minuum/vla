import json
import glob
from pathlib import Path

configs = sorted(glob.glob("Mobile_VLA/configs/mobile_vla_v3_exp0*.json"))
for c in configs:
    with open(c, 'r') as f:
        data = json.load(f)
        
    print(f"--- {Path(c).name} ---")
    print(f"Exp Name:     {data.get('exp_name')}")
    print(f"Max Epochs:   {data.get('max_epochs')}")
    print(f"Batch Size:   {data.get('batch_size')}")
    
    train_setup = data.get('train_setup', {})
    print(f"LoRA Enable:  {train_setup.get('lora_enable')}")
    print(f"LoRA Rank:    {train_setup.get('lora_r')}")
    
    act_head = data.get('act_head', {})
    print(f"Num Classes:  {act_head.get('num_classes')}")
    print(f"Class Wgts:   {act_head.get('class_weights')}")
    
    train_dataset = data.get('train_dataset', {})
    print(f"Filter KW:    {train_dataset.get('episode_filter_keyword', 'None')}")
    print(f"Color Jitter: {train_dataset.get('use_color_jitter')}")
    print(f"Random Crop:  {train_dataset.get('use_random_crop')}")
    print(f"History DO:   {train_dataset.get('history_dropout_prob', 'None')}")
    print(f"Comment:      {data.get('_comment', 'None')}")
    print()
