#!/usr/bin/env python3
"""
Pretrained VLM 로드 테스트
Config의 load_vlm_only 기능 검증
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'RoboVLMs_upstream'))

from robovlms.train.mobile_vla_trainer import MobileVLATrainer
from robovlms.utils.config_utils import load_config


def main():
    print("=" * 70)
    print(" Pretrained VLM 로드 테스트")
    print("=" * 70)
    
    config_path = "Mobile_VLA/configs/mobile_vla_pretrained.json"
    
    print(f"\n[Config] {config_path}")
    configs = load_config(config_path)
    
    # 핵심 설정 확인
    print(f"\n[핵심 설정]")
    print(f"  pretrained_vlm_path: {configs.get('pretrained_vlm_path', 'N/A')}")
    print(f"  load_vlm_only: {configs.get('load_vlm_only', 'N/A')}")
    print(f"  freeze_backbone: {configs.get('train_setup', {}).get('freeze_backbone', 'N/A')}")
    print(f"  action_dim: {configs.get('act_head', {}).get('action_dim', 'N/A')}")
    
    # Pretrained 파일 확인
    pretrained_path = configs.get('pretrained_vlm_path')
    if pretrained_path and os.path.exists(pretrained_path):
        file_size = os.path.getsize(pretrained_path) / (1024**3)
        print(f"  ✅ Pretrained file exists: {file_size:.2f} GB")
    else:
        print(f"  ❌ Pretrained file not found!")
        return
    
    # 모델 초기화 (pretrained VLM 로드)
    print(f"\n[모델 초기화 중...] (Pretrained VLM 로드)")
    print("(약 2분 소요)\n")
    
    model = MobileVLATrainer(configs)
    
    # 파라미터 상태 확인
    print("\n" + "=" * 70)
    print(" 파라미터 상태 확인")
    print("=" * 70)
    
    total_params = 0
    trainable_params = 0
    frozen_params = 0
    
    vlm_trainable = 0
    vlm_frozen = 0
    action_head_trainable = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            if 'act_head' in name or 'action_token' in name:
                action_head_trainable += param.numel()
        else:
            frozen_params += param.numel()
            if 'act_head' not in name:
                vlm_frozen += param.numel()
    
    print(f"\n  총 파라미터:         {total_params:>15,}")
    print(f"  Trainable:           {trainable_params:>15,} ({100*trainable_params/total_params:.2f}%)")
    print(f"  Frozen:              {frozen_params:>15,} ({100*frozen_params/total_params:.2f}%)")
    print(f"\n  VLM (Frozen):        {vlm_frozen:>15,}")
    print(f"  Action Head (Train): {action_head_trainable:>15,}")
    
    # 성공 확인
    print("\n" + "=" * 70)
    print(" 결과")
    print("=" * 70)
    
    if vlm_frozen > 0 and action_head_trainable > 0:
        print(f"\n  ✅ SUCCESS: Pretrained VLM 로드 완료!")
        print(f"  ✅ VLM이 Frozen 상태로 로드됨")
        print(f"  ✅ Action Head (2DoF)가 새로 초기화되어 학습 가능")
    else:
        print(f"\n  ⚠️ 확인 필요: 파라미터 상태 점검 바람")


if __name__ == "__main__":
    main()
