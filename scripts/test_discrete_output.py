import os
import sys
import json
import torch

import sys
import sys
import os
sys.path.insert(0, os.path.abspath('third_party/RoboVLMs'))

from robovlms.train.mobile_vla_trainer import MobileVLATrainer
from robovlms.data.mobile_vla_h5_dataset import MobileVLAH5Dataset
from robovlms.utils.model_utils import build_tokenizer

# Patches required for RoboVLMs 
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

ckpt = "/home/billy/25-1kp/vla/runs/v3_classification/kosmos/mobile_vla_v3_exp08_center_goal/2026-03-05/v3-exp08-center-goal/epoch_epoch=07-val_loss=val_loss=0.031.ckpt"
cfg = "/home/billy/25-1kp/vla/configs/mobile_vla_v3_exp08_center_goal.json"

with open(cfg, 'r') as f: config = json.load(f)

# Need to inject model backbone
from robovlms.model.backbone.robokosmos import RoboKosMos
import robovlms.model.backbone as backbone
backbone.__dict__["RoboVLM-Nav"] = RoboKosMos

trainer = MobileVLATrainer.load_from_checkpoint(ckpt, config_path=cfg, map_location="cuda")
model = trainer.model.to('cuda')
model.eval()

tokenizer_config = config.get('tokenizer', None)
tokenizer = build_tokenizer(tokenizer_config) if tokenizer_config else None
dt = tokenizer.tokenizer if hasattr(tokenizer, 'tokenizer') else tokenizer

ds = MobileVLAH5Dataset(
    data_dir="/home/billy/25-1kp/vla/ROS_action/basket_dataset_v2",
    episode_pattern="episode_*.h5",
    model_name="kosmos",
    train_split=0.0,
    is_validation=True,
    window_size=8,
    fwd_pred_next_n=1,
    tokenizer=dt,
    tokenizer_config=tokenizer_config,
    discrete_action=True,
    instruction_preset="center_goal"
)

sample = ds[0]
batch = ds.collater([sample])
gpu_batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

with torch.no_grad():
    prediction = model.inference(
        gpu_batch['rgb'],
        gpu_batch['text'],
        attention_mask=gpu_batch['text_mask'],
        vision_gripper=gpu_batch['hand_rgb'],
        raw_text=gpu_batch['raw_text'],
        data_source=gpu_batch['data_source']
    )

    logits = prediction['action']
    if isinstance(logits, tuple): logits = logits[0]
    print(f"Prediction type: {type(logits)}")
    print(f"Prediction shape: {logits.shape}")
    print(f"GT shape: {gpu_batch['action_chunck'].shape}")
    print(f"GT content: {gpu_batch['action_chunck']}")
    
    pred_class = logits.argmax(dim=-1)
    print(f"Predict class shape: {pred_class.shape}")
    print(f"Predict class: {pred_class}")
