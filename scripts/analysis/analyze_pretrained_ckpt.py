#!/usr/bin/env python3
"""
RoboVLMs Pretrained Checkpoint 구조 분석
Google Robot pretrained checkpoint의 구조를 분석하고
Mobile VLA (2DoF)로 전이학습 가능성 확인
"""

import os
import sys
import torch

def print_section(title, char="="):
    print(f"\n{char * 70}")
    print(f" {title}")
    print(f"{char * 70}")


def main():
    print_section("RoboVLMs Pretrained Checkpoint 분석")
    
    ckpt_path = "pretrained_ckpts/checkpoints/kosmos_ph_google-robot-post-train.pt"
    
    if not os.path.exists(ckpt_path):
        print(f"❌ Checkpoint not found: {ckpt_path}")
        return
    
    print(f"\n[Checkpoint] {ckpt_path}")
    print(f"[Size] {os.path.getsize(ckpt_path) / (1024**3):.2f} GB")
    
    # Checkpoint 로드
    print("\n로딩 중... (약 1분 소요)")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    
    # Checkpoint 구조 분석
    print_section("Checkpoint 구조", "-")
    
    if isinstance(checkpoint, dict):
        print(f"\n[Top-level keys]")
        for k in checkpoint.keys():
            v = checkpoint[k]
            if isinstance(v, torch.Tensor):
                print(f"  {k}: Tensor {v.shape}")
            elif isinstance(v, dict):
                print(f"  {k}: dict ({len(v)} keys)")
            else:
                print(f"  {k}: {type(v).__name__}")
    
    # state_dict 분석
    state_dict = checkpoint.get('state_dict', checkpoint)
    
    print_section("파라미터 분류", "-")
    
    # 구성 요소별 분류
    components = {
        "VLM (Backbone)": [],
        "Vision Encoder": [],
        "Text Encoder": [],
        "Action Head": [],
        "Other": []
    }
    
    for name, param in state_dict.items():
        if isinstance(param, torch.Tensor):
            if 'vision_model' in name or 'vision_encoder' in name:
                components["Vision Encoder"].append((name, param.shape, param.numel()))
            elif 'text_model' in name or 'language_model' in name or 'text_encoder' in name:
                components["Text Encoder"].append((name, param.shape, param.numel()))
            elif 'act_head' in name or 'action_head' in name or 'policy_head' in name:
                components["Action Head"].append((name, param.shape, param.numel()))
            elif 'backbone' in name:
                components["VLM (Backbone)"].append((name, param.shape, param.numel()))
            else:
                components["Other"].append((name, param.shape, param.numel()))
    
    # 요약 출력
    print(f"\n{'구성 요소':<25} {'파라미터 수':>15} {'레이어 수':>10}")
    print("-" * 55)
    
    total_params = 0
    action_head_params = 0
    
    for comp, params in components.items():
        num_params = sum(p[2] for p in params)
        total_params += num_params
        if comp == "Action Head":
            action_head_params = num_params
        print(f"{comp:<25} {num_params:>15,} {len(params):>10}")
    
    print("-" * 55)
    print(f"{'Total':<25} {total_params:>15,}")
    
    # Action Head 상세
    print_section("Action Head 상세 (7DoF)", "-")
    
    for name, shape, numel in components["Action Head"][:10]:  # 처음 10개만
        print(f"  {name}")
        print(f"    Shape: {shape}")
    
    if len(components["Action Head"]) > 10:
        print(f"  ... 외 {len(components['Action Head']) - 10}개")
    
    # 전이학습 가능성 분석
    print_section("전이학습 분석 (7DoF → 2DoF)")
    
    print(f"""
  [현재 상태]
  - Pretrained: Google Robot (7DoF arm action)
  - Action Head 파라미터: {action_head_params:,}
  
  [목표]
  - Mobile VLA: 2DoF velocity (linear_x, linear_y)
  
  [전이학습 전략]
  
  ✅ 방법 1: VLM만 재사용 (권장)
    - Pretrained VLM (Vision + Text Encoder) 로드
    - Action Head 새로 초기화 (7DoF → 2DoF)
    - Action Head만 학습
    
  ⚠️ 방법 2: 전체 Fine-tuning
    - Pretrained 전체 로드
    - Action Head 교체 (7DoF → 2DoF)
    - LoRA로 VLM도 함께 학습
    
  ❌ 방법 3: 직접 사용 (불가)
    - 7DoF와 2DoF는 Action space가 다름
    - 직접 사용 불가능
""")
    
    # VLM 파라미터 키 출력 (재사용 가능)
    print_section("재사용 가능한 VLM 키 예시 (처음 5개)")
    
    vlm_keys = [k for k in state_dict.keys() if 'vision_model' in k or 'text_model' in k or 'backbone' in k]
    for k in vlm_keys[:5]:
        print(f"  {k}")
    
    if len(vlm_keys) > 5:
        print(f"  ... 외 {len(vlm_keys) - 5}개")


if __name__ == "__main__":
    main()
