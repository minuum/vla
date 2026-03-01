#!/usr/bin/env python3
"""
V3-EXP-05 LoRA 머지 스크립트 (Step 4)
학습 완료 후 best checkpoint를 LoRA merge하여 standalone 모델 생성

사용법:
  python3 merge_lora_exp05.py                         # best ckpt 자동 탐색
  python3 merge_lora_exp05.py --ckpt_path <path>      # 특정 ckpt 지정
"""

import argparse
import json
import torch
import os
import sys
import glob

sys.path.append(os.path.join('/home/billy/25-1kp/vla/RoboVLMs_upstream'))

RUNS_DIR = '/home/billy/25-1kp/vla/RoboVLMs_upstream/runs/v3_classification/kosmos/mobile_vla_v3_exp05_lora'
OUT_MERGED = '/home/billy/25-1kp/vla/v3-exp05-lora/merged_v3_exp05_best.ckpt'
CONFIG_EXP05 = '/home/billy/25-1kp/vla/Mobile_VLA/configs/mobile_vla_v3_exp05_lora.json'
CONFIG_BASE  = '/home/billy/25-1kp/vla/Mobile_VLA/configs/mobile_vla_v3_exp01_aug.json'


def find_best_ckpt():
    """val_loss가 가장 낮은 checkpoint 자동 탐색"""
    pattern = os.path.join(RUNS_DIR, '**', 'epoch_*.ckpt')
    ckpts = glob.glob(pattern, recursive=True)
    if not ckpts:
        raise FileNotFoundError(f'No checkpoints found in {RUNS_DIR}')

    best_ckpt = None
    best_loss = float('inf')
    for ckpt in ckpts:
        # 파일명에서 val_loss 파싱
        # 예: epoch_epoch=05-val_loss=val_loss=0.234.ckpt
        if 'val_loss=' in ckpt:
            try:
                val_str = ckpt.split('val_loss=val_loss=')[-1].replace('.ckpt', '')
                val_loss = float(val_str)
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_ckpt = ckpt
            except ValueError:
                continue

    if best_ckpt is None:
        # val_loss 없으면 most recent
        best_ckpt = max(ckpts, key=os.path.getmtime)
        print(f'⚠️ val_loss 파싱 실패, 가장 최신 ckpt 선택: {best_ckpt}')
    else:
        print(f'✅ Best ckpt: {best_ckpt} (val_loss={best_loss:.4f})')

    return best_ckpt


def get_configs():
    """exp05 config 로드 (base + exp05 override)"""
    with open(CONFIG_BASE, 'r') as f:
        config_dict = json.load(f)
    with open(CONFIG_EXP05, 'r') as f:
        exp05 = json.load(f)
        for k, v in exp05.items():
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

    vlm = config_dict.get('vlm')
    if vlm and not os.path.exists(vlm.get('pretrained_model_name_or_path', '')):
        vlm['pretrained_model_name_or_path'] = 'microsoft/kosmos-2-patch14-224'

    config_dict['tokenizer'] = {
        'type': 'AutoProcessor',
        'pretrained_model_name_or_path': 'microsoft/kosmos-2-patch14-224',
        'tokenizer_type': 'kosmos'
    }
    return config_dict


def merge_checkpoint(ckpt_path, out_path):
    from peft import PeftModel

    print(f'🔧 모델 초기화 중...')
    try:
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

    print(f'📦 Checkpoint 로드: {ckpt_path}')
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    state_dict = checkpoint['state_dict']
    unwrapped_sd = {}
    for k, v in state_dict.items():
        unwrapped_sd[k.replace('model.', '', 1) if k.startswith('model.') else k] = v

    missing, unexpected = model.load_state_dict(unwrapped_sd, strict=False)
    print(f'  Missing: {len(missing)}, Unexpected: {len(unexpected)}')
    if missing:
        act_missing = [k for k in missing if 'act_head' in k]
        if act_missing:
            print(f'  ❌ CRITICAL: act_head 일부 없음: {act_missing[:3]}')

    print('🔗 LoRA weights 머지 중...')
    if hasattr(model, 'backbone'):
        for part in ['text_model', 'vision_model']:
            sub = getattr(model.backbone, part, None)
            if sub and isinstance(sub, PeftModel):
                print(f'  Merging {part}...')
                setattr(model.backbone, part, sub.merge_and_unload())

    # state_dict 저장
    sd_to_save = {f'model.{k}': v for k, v in model.state_dict().items()}
    checkpoint['state_dict'] = sd_to_save

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(checkpoint, out_path)
    print(f'✅ 저장 완료: {out_path}')
    print(f'  파일 크기: {os.path.getsize(out_path)/1024**3:.2f} GB')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default=None, help='특정 ckpt 경로 (없으면 best 자동 탐색)')
    parser.add_argument('--out_path', type=str, default=OUT_MERGED)
    args = parser.parse_args()

    ckpt = args.ckpt_path or find_best_ckpt()
    merge_checkpoint(ckpt, args.out_path)
