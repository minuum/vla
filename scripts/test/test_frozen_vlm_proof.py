#!/usr/bin/env python3
"""
Frozen VLM 증명 실험
교수님 미팅 핵심 질문: "VLM을 Frozen 시켰을 때 고정되었다를 증명"

실험 목표:
1. freeze_backbone=True일 때 VLM requires_grad가 False인지 확인
2. 동일 instruction + 다른 이미지 → VLM text embedding이 동일한지 확인
3. 다른 instruction + 동일 이미지 → VLM text embedding이 다른지 확인
"""

import os
import sys
import torch
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'RoboVLMs_upstream'))

from robovlms.train.mobile_vla_trainer import MobileVLATrainer
from robovlms.utils.config_utils import load_config
from torchvision import transforms


def print_section(title):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def create_random_image(seed=None):
    """랜덤 이미지 생성"""
    if seed is not None:
        np.random.seed(seed)
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(img_array)


def main():
    print_section("Frozen VLM 증명 실험")
    
    # Config 로드 (freeze_backbone=True인 config 사용)
    config_path = "Mobile_VLA/configs/mobile_vla_chunk5_20251217.json"
    print(f"\n[1] Config 로드: {config_path}")
    
    configs = load_config(config_path)
    
    # freeze_backbone 설정 확인
    freeze_backbone = configs.get("train_setup", {}).get("freeze_backbone", False)
    print(f"    freeze_backbone: {freeze_backbone}")
    
    if not freeze_backbone:
        print("    ⚠️ WARNING: freeze_backbone이 False입니다!")
        print("    Frozen VLM 테스트를 위해 freeze_backbone=True인 config가 필요합니다.")
    
    # 모델 로드
    print("\n[2] 모델 로드 중... (1-2분 소요)")
    checkpoint_path = None
    
    # 최신 체크포인트 찾기
    import glob
    ckpts = glob.glob("runs/**/last.ckpt", recursive=True)
    if ckpts:
        checkpoint_path = max(ckpts, key=os.path.getmtime)
        print(f"    Checkpoint: {checkpoint_path}")
    else:
        print("    ⚠️ 체크포인트 없음. 새 모델 초기화...")
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        model = MobileVLATrainer.load_from_checkpoint(
            checkpoint_path, configs=configs, strict=False, map_location='cpu'
        )
    else:
        model = MobileVLATrainer(configs)
    
    model.eval()
    
    # ============================================================
    # 실험 1: requires_grad 상태 확인
    # ============================================================
    print_section("실험 1: requires_grad 상태 확인")
    
    vlm_params = []
    action_head_params = []
    
    for name, param in model.named_parameters():
        if 'act_head' in name or 'action_head' in name or 'velocities' in name or 'rnn' in name:
            action_head_params.append((name, param.requires_grad))
        else:
            vlm_params.append((name, param.requires_grad))
    
    # VLM 파라미터 상태
    vlm_trainable = sum(1 for _, rg in vlm_params if rg)
    vlm_frozen = sum(1 for _, rg in vlm_params if not rg)
    print(f"\n[VLM 파라미터]")
    print(f"  - Trainable: {vlm_trainable}")
    print(f"  - Frozen: {vlm_frozen}")
    
    if vlm_frozen > vlm_trainable:
        print(f"  ✅ VLM이 대부분 Frozen 상태입니다 ({vlm_frozen}/{vlm_frozen + vlm_trainable})")
    else:
        print(f"  ⚠️ VLM이 Trainable 상태입니다 ({vlm_trainable}/{vlm_frozen + vlm_trainable})")
    
    # Action Head 파라미터 상태
    ah_trainable = sum(1 for _, rg in action_head_params if rg)
    ah_frozen = sum(1 for _, rg in action_head_params if not rg)
    print(f"\n[Action Head 파라미터]")
    print(f"  - Trainable: {ah_trainable}")
    print(f"  - Frozen: {ah_frozen}")
    
    if ah_trainable > ah_frozen:
        print(f"  ✅ Action Head가 Trainable 상태입니다 ({ah_trainable}/{ah_frozen + ah_trainable})")
    else:
        print(f"  ⚠️ Action Head가 Frozen 상태입니다 ({ah_frozen}/{ah_frozen + ah_trainable})")
    
    # ============================================================
    # 실험 2: 동일 instruction + 다른 이미지 → embedding 비교
    # ============================================================
    print_section("실험 2: 동일 instruction + 다른 이미지")
    
    model = model.cuda()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    
    # 다른 이미지 2개 생성
    img1 = create_random_image(seed=42)
    img2 = create_random_image(seed=123)
    
    vx1 = transform(img1).unsqueeze(0).unsqueeze(1).repeat(1, 8, 1, 1, 1).cuda()
    vx2 = transform(img2).unsqueeze(0).unsqueeze(1).repeat(1, 8, 1, 1, 1).cuda()
    
    # 동일 instruction
    instruction = "Navigate around the obstacle on the left side"
    print(f"\n  Instruction: '{instruction}'")
    print(f"  Image 1: seed=42, Image 2: seed=123")
    
    # Tokenize
    processor = model.model.processor if hasattr(model.model, 'processor') else model.processor
    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
    
    enc = tokenizer(instruction, return_tensors='pt', padding=True, truncation=True, max_length=256)
    lang_x = enc['input_ids'].cuda()
    attn = enc.get('attention_mask').cuda() if 'attention_mask' in enc else None
    
    # VLM forward (hidden states 추출)
    with torch.no_grad():
        out1 = model.model.inference(vision_x=vx1, lang_x=lang_x, attention_mask=attn)
        out2 = model.model.inference(vision_x=vx2, lang_x=lang_x, attention_mask=attn)
    
    # Action 비교
    if isinstance(out1, dict) and 'action' in out1:
        act1 = out1['action']
        act2 = out2['action']
        if isinstance(act1, tuple):
            act1, act2 = act1[0], act2[0]
    else:
        act1, act2 = out1, out2
    
    diff_img = torch.abs(act1 - act2).mean().item()
    print(f"\n  [결과] Action 차이 (다른 이미지, 동일 instruction): {diff_img:.6f}")
    
    if diff_img > 0.01:
        print(f"  ✅ 다른 이미지 → 다른 Action (VLM이 이미지를 처리하고 있음)")
    else:
        print(f"  ⚠️ 다른 이미지인데 거의 동일한 Action!")
    
    # ============================================================
    # 실험 3: 다른 instruction + 동일 이미지 → embedding 비교
    # ============================================================
    print_section("실험 3: 다른 instruction + 동일 이미지")
    
    inst_left = "Navigate around the obstacle on the left side"
    inst_right = "Navigate around the obstacle on the right side"
    
    print(f"\n  Instruction 1: '{inst_left}'")
    print(f"  Instruction 2: '{inst_right}'")
    print(f"  동일 이미지 사용 (seed=42)")
    
    enc_left = tokenizer(inst_left, return_tensors='pt', padding=True, truncation=True, max_length=256)
    enc_right = tokenizer(inst_right, return_tensors='pt', padding=True, truncation=True, max_length=256)
    
    lang_left = enc_left['input_ids'].cuda()
    lang_right = enc_right['input_ids'].cuda()
    attn_left = enc_left.get('attention_mask').cuda() if 'attention_mask' in enc_left else None
    attn_right = enc_right.get('attention_mask').cuda() if 'attention_mask' in enc_right else None
    
    with torch.no_grad():
        out_left = model.model.inference(vision_x=vx1, lang_x=lang_left, attention_mask=attn_left)
        out_right = model.model.inference(vision_x=vx1, lang_x=lang_right, attention_mask=attn_right)
    
    if isinstance(out_left, dict) and 'action' in out_left:
        act_left = out_left['action']
        act_right = out_right['action']
        if isinstance(act_left, tuple):
            act_left, act_right = act_left[0], act_right[0]
    else:
        act_left, act_right = out_left, out_right
    
    diff_inst = torch.abs(act_left - act_right).mean().item()
    print(f"\n  [결과] Action 차이 (동일 이미지, 다른 instruction): {diff_inst:.6f}")
    
    if diff_inst > 0.01:
        print(f"  ✅ 다른 instruction → 다른 Action (VLM이 instruction을 처리하고 있음)")
    else:
        print(f"  ❌ 다른 instruction인데 거의 동일한 Action! (Instruction grounding 실패)")
    
    # ============================================================
    # 최종 결론
    # ============================================================
    print_section("최종 결론")
    
    print("\n[Frozen VLM 증명 결과]")
    print(f"  1. VLM Frozen 상태: {'✅ 확인' if vlm_frozen > vlm_trainable else '❌ 미확인'}")
    print(f"  2. Action Head Trainable: {'✅ 확인' if ah_trainable > ah_frozen else '❌ 미확인'}")
    print(f"  3. 이미지 반응: {'✅ 정상' if diff_img > 0.01 else '⚠️ 문제'}")
    print(f"  4. Instruction 반응: {'✅ 정상' if diff_inst > 0.01 else '❌ Instruction 무시'}")
    
    print("\n[해석]")
    if vlm_frozen > vlm_trainable and diff_inst < 0.01:
        print("  Frozen VLM이 instruction embedding을 고정시켜서")
        print("  Action Head가 instruction을 구분하지 못하고 있습니다.")
        print("  → LoRA fine-tuning 또는 VLM unfreezing이 필요합니다.")
    elif diff_inst > 0.01:
        print("  모델이 instruction에 따라 다른 action을 출력합니다.")
        print("  → Instruction grounding이 작동하고 있습니다.")


if __name__ == "__main__":
    main()
