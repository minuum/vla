#!/usr/bin/env python3
"""
Pretrained VLM Checkpoint 직접 로드 테스트
HuggingFace 다운로드 없이 checkpoint에서 직접 로드
"""

import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'RoboVLMs_upstream'))


def print_section(title, char="="):
    print(f"\n{char * 70}")
    print(f" {title}")
    print(f"{char * 70}")


def main():
    print_section("Pretrained VLM Checkpoint 직접 로드 테스트")
    
    ckpt_path = "pretrained_ckpts/checkpoints/kosmos_ph_google-robot-post-train.pt"
    
    if not os.path.exists(ckpt_path):
        print(f"❌ Checkpoint not found: {ckpt_path}")
        return
    
    print(f"\n[Checkpoint] {ckpt_path}")
    print(f"[Size] {os.path.getsize(ckpt_path) / (1024**3):.2f} GB")
    
    # Checkpoint 로드
    print("\n로딩 중... (약 1분 소요)")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    
    # State dict 추출
    state_dict = checkpoint.get('state_dict', checkpoint)
    
    # 구성 요소별 분류
    vlm_weights = {}
    action_head_weights = {}
    other_weights = {}
    
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            # "model." prefix 제거 (있다면)
            clean_key = k.replace("model.", "") if k.startswith("model.") else k
            
            if 'act_head' in k or 'action_token' in k:
                action_head_weights[k] = v
            elif 'backbone' in k or 'vision_model' in k or 'text_model' in k:
                vlm_weights[clean_key] = v
            else:
                other_weights[k] = v
    
    print(f"\n[분석 결과]")
    print(f"  VLM weights: {len(vlm_weights)}")
    print(f"  Action Head weights: {len(action_head_weights)}")
    print(f"  Other weights: {len(other_weights)}")
    
    # VLM 파라미터 수 계산
    vlm_params = sum(v.numel() for v in vlm_weights.values())
    ah_params = sum(v.numel() for v in action_head_weights.values())
    
    print(f"\n[파라미터 수]")
    print(f"  VLM: {vlm_params:,} ({vlm_params/1e9:.2f}B)")
    print(f"  Action Head: {ah_params:,} ({ah_params/1e6:.1f}M)")
    
    # VLM Key 예시
    print_section("VLM 키 구조 (처음 10개)")
    for i, k in enumerate(list(vlm_weights.keys())[:10]):
        print(f"  {k}")
    
    # Action Head Key
    print_section("Action Head 키 (교체 대상)")
    for k in action_head_weights.keys():
        print(f"  ❌ {k} (7DoF → 2DoF 교체 필요)")
    
    # 저장 방법 제안
    print_section("전이학습 실행 방법")
    print("""
  [방법 1] 기존 학습된 checkpoint에서 VLM만 추출
    - VLM weights만 저장 후 새 학습에 로드
    
  [방법 2] 학습 시작 시 자동 로드 (현재 구현됨)
    - base_backbone.py에서 pretrained_vlm_path 설정 시 자동 로드
    - Action Head는 새로 초기화됨
    
  [권장] 이미 학습된 chunk5 checkpoint에서 VLM을 추출하여 사용
""")


if __name__ == "__main__":
    main()
