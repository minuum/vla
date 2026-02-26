import argparse
import json
import torch
import os
import sys

from peft import PeftModel
from omegaconf import OmegaConf

sys.path.append(os.path.join('/home/billy/25-1kp/vla/RoboVLMs_upstream'))

def get_configs():
    with open('/home/billy/25-1kp/vla/Mobile_VLA/configs/mobile_vla_v3_exp01_aug.json', 'r') as f:
        config_dict = json.load(f)
    with open('/home/billy/25-1kp/vla/Mobile_VLA/configs/mobile_vla_v3_exp04_lora.json', 'r') as f:
        exp04 = json.load(f)
        for k, v in exp04.items():
            if isinstance(v, dict) and k in config_dict and isinstance(config_dict[k], dict):
                config_dict[k].update(v)
            else:
                config_dict[k] = v
        
    config_dict['use_hand_rgb'] = config_dict.get('use_hand_rgb', False)
    config_dict['fwd_pred_next_n'] = config_dict.get('act_head', {}).get('fwd_pred_next_n', 1)
    config_dict['fwd_head'] = config_dict.get('fwd_head', {})
    config_dict['window_size'] = config_dict.get('act_head', {}).get('window_size', 8)
    config_dict['cap_loss_ratio'] = config_dict.get('cap_loss_ratio', 1.0)
    config_dict['arm_gripper_loss_ratio'] = config_dict.get('arm_gripper_loss_ratio', 1.0)
    config_dict['fwd_loss_ratio'] = config_dict.get('fwd_loss_ratio', 1.0)
    
    # ensure model exists
    vlm = config_dict.get('vlm')
    if vlm and not os.path.exists(vlm.get('pretrained_model_name_or_path', '')):
        print(f'Warning: VLM path {vlm.get("pretrained_model_name_or_path")} does not exist, replacing with Microsoft/kosmos-2-patch14-224')
        vlm['pretrained_model_name_or_path'] = 'microsoft/kosmos-2-patch14-224'
    
    config_dict['tokenizer'] = {'type': 'AutoProcessor', 'pretrained_model_name_or_path': 'microsoft/kosmos-2-patch14-224', 'tokenizer_type': 'kosmos'}
        
    return config_dict

def merge_checkpoint(ckpt_path, out_path):
    print(f'Creating model...')
    try:
        # patch default_tokenizer_config dynamically
        import robovlms.utils.model_utils
        orig_dtc = robovlms.utils.model_utils.default_tokenizer_config
        def new_dtc(tokenizer):
            if tokenizer == 'kosmos':
                return {'type': 'AutoProcessor', 'pretrained_model_name_or_path': 'microsoft/kosmos-2-patch14-224', 'tokenizer_type': 'kosmos'}
            try:
                return orig_dtc(tokenizer)
            except Exception:
                return {'type': 'AutoProcessor', 'pretrained_model_name_or_path': 'microsoft/kosmos-2-patch14-224', 'tokenizer_type': 'kosmos'}
        robovlms.utils.model_utils.default_tokenizer_config = new_dtc

        from robovlms.train.mobile_vla_trainer import MobileVLATrainer
        config = get_configs()
        trainer = MobileVLATrainer(config)
        model = trainer.model
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise e

    print('Is model loaded?', model is not None)

    checkpoint = torch.load(ckpt_path, map_location='cpu')

    state_dict = checkpoint['state_dict']
    unwrapped_sd = {}
    for k, v in state_dict.items():
        if k.startswith('model.'):
            unwrapped_sd[k.replace('model.', '')] = v
        else:
            unwrapped_sd[k] = v

    print('Loading state dict...')
    model.load_state_dict(unwrapped_sd, strict=False)

    print('Merging LoRA...')
    if hasattr(model, 'backbone') and hasattr(model.backbone, 'text_model') and isinstance(model.backbone.text_model, PeftModel):
        print('Merging text_model...')
        model.backbone.text_model = model.backbone.text_model.merge_and_unload()
    if hasattr(model, 'backbone') and hasattr(model.backbone, 'vision_model') and isinstance(model.backbone.vision_model, PeftModel):
        print('Merging vision_model...')
        model.backbone.vision_model = model.backbone.vision_model.merge_and_unload()
        
    sd_to_save = {}
    for k, v in model.state_dict().items():
        sd_to_save['model.' + k] = v
        
    checkpoint['state_dict'] = sd_to_save
    torch.save(checkpoint, out_path)
    print('Saved to', out_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default='/home/billy/25-1kp/vla/RoboVLMs_upstream/runs/v3_classification/kosmos/mobile_vla_v3_exp04_lora/2026-02-22/v3-exp04-lora/epoch_epoch=05-val_loss=val_loss=0.234.ckpt')
    parser.add_argument('--out_path', type=str, default='/home/billy/25-1kp/vla/RoboVLMs_upstream/runs/v3_classification/kosmos/mobile_vla_v3_exp04_lora/2026-02-22/v3-exp04-lora/merged_v3_exp04_best.ckpt')
    args = parser.parse_args()
    merge_checkpoint(args.ckpt_path, args.out_path)
