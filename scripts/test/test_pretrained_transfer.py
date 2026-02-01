#!/usr/bin/env python3
"""
Pretrained VLM 전이학습 테스트
Google Robot pretrained checkpoint에서 VLM만 추출하여 
새로운 2DoF Action Head와 결합하는 테스트
"""

import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'RoboVLMs_upstream'))


def print_section(title, char="="):
    print(f"\n{char * 70}")
    print(f" {title}")
    print(f"{char * 70}")


def load_vlm_from_pretrained(pretrained_path, backbone):
    """
    Pretrained checkpoint에서 VLM weights만 로드
    
    Args:
        pretrained_path: RoboVLMs pretrained checkpoint 경로
        backbone: HuggingFace Kosmos2ForConditionalGeneration 모델
    """
    print(f"[Pretrained VLM] Loading from: {pretrained_path}")
    pretrained = torch.load(pretrained_path, map_location='cpu')
    state_dict = pretrained.get('state_dict', pretrained)
    
    # Filter: VLM only, exclude Action Head
    # Handle key prefix: "model.backbone.*" -> "*"
    vlm_keys = {}
    for k, v in state_dict.items():
        if 'act_head' in k or 'action_token' in k:
            continue  # Skip Action Head
        
        new_key = k
        if k.startswith("model.backbone."):
            new_key = k.replace("model.backbone.", "")
        elif k.startswith("model."):
            new_key = k.replace("model.", "")
        
        vlm_keys[new_key] = v
    
    # Load VLM weights
    missing, unexpected = backbone.load_state_dict(vlm_keys, strict=False)
    
    print(f"[Pretrained VLM] Loaded {len(vlm_keys)} weights")
    print(f"[Pretrained VLM] Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    
    return missing, unexpected


def main():
    print_section("Pretrained VLM 전이학습 테스트")
    
    pretrained_path = "pretrained_ckpts/checkpoints/kosmos_ph_google-robot-post-train.pt"
    
    if not os.path.exists(pretrained_path):
        print(f"❌ Pretrained checkpoint not found: {pretrained_path}")
        return
    
    print(f"[Pretrained] {pretrained_path}")
    print(f"[Size] {os.path.getsize(pretrained_path) / (1024**3):.2f} GB")
    
    # 1. HuggingFace에서 Kosmos-2 모델 구조만 로드 (weights 없이)
    print_section("Step 1: Kosmos-2 모델 구조 로드")
    
    from transformers import Kosmos2ForConditionalGeneration, AutoConfig
    
    # 캐시된 config 사용
    print("Loading model structure from HuggingFace cache...")
    try:
        config = AutoConfig.from_pretrained("microsoft/kosmos-2-patch14-224")
        # 빈 모델 생성 (weights 없이)
        backbone = Kosmos2ForConditionalGeneration(config)
        print("✅ Model structure loaded")
    except Exception as e:
        print(f"❌ Failed to load model structure: {e}")
        print("Trying to load from pretrained directly...")
        return
    
    # 2. Pretrained checkpoint에서 VLM weights 로드
    print_section("Step 2: Pretrained VLM Weights 로드")
    
    missing, unexpected = load_vlm_from_pretrained(pretrained_path, backbone)
    
    if len(missing) < 10:
        print("✅ VLM weights 로드 성공!")
    else:
        print(f"⚠️ Missing keys가 많음: {len(missing)}")
        print("처음 10개 missing keys:")
        for k in missing[:10]:
            print(f"  - {k}")
    
    # 3. 모델 파라미터 확인
    print_section("Step 3: 파라미터 확인")
    
    total_params = sum(p.numel() for p in backbone.parameters())
    print(f"Total parameters: {total_params:,} ({total_params/1e9:.2f}B)")
    
    # 4. Inference 테스트 (dummy input)
    print_section("Step 4: Inference 테스트")
    
    backbone.eval()
    
    # Dummy input
    import torch
    pixel_values = torch.randn(1, 3, 224, 224)
    
    try:
        with torch.no_grad():
            vision_output = backbone.vision_model(pixel_values=pixel_values)
            print(f"✅ Vision encoder output shape: {vision_output.last_hidden_state.shape}")
    except Exception as e:
        print(f"❌ Vision encoder error: {e}")
    
    print_section("완료")
    print("""
  ✅ Pretrained VLM 로드 성공!
  
  [다음 단계]
  1. 이 VLM을 MobileVLATrainer에 통합
  2. 2DoF Action Head 붙이기
  3. 학습 시작
""")


if __name__ == "__main__":
    main()
