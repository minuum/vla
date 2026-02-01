#!/usr/bin/env python3
"""
Kosmos-2 모델 구조 상세 분석
어떤 부분이 Frozen되고 어떤 부분이 Trainable인지 시각화
"""

import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'RoboVLMs_upstream'))

from robovlms.train.mobile_vla_trainer import MobileVLATrainer
from robovlms.utils.config_utils import load_config
import glob


def print_section(title, char="="):
    print(f"\n{char * 70}")
    print(f" {title}")
    print(f"{char * 70}")


def count_params(module):
    """모듈의 파라미터 수 계산"""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def main():
    print_section("Kosmos-2 모델 구조 상세 분석")
    
    # Config 로드
    config_path = "Mobile_VLA/configs/mobile_vla_chunk5_20251217.json"
    print(f"\n[Config] {config_path}")
    
    configs = load_config(config_path)
    
    # 핵심 설정 출력
    train_setup = configs.get("train_setup", {})
    print_section("핵심 학습 설정", "-")
    print(f"""
  freeze_backbone:     {train_setup.get('freeze_backbone', 'N/A')}
  train_vision:        {train_setup.get('train_vision', 'N/A')}
  train_text_embedding: {train_setup.get('train_text_embedding', 'N/A')}
  lora_enable:         {train_setup.get('lora_enable', 'N/A')}
  predict_action:      {train_setup.get('predict_action', 'N/A')}
""")
    
    # Instruction 설정 확인
    print_section("Instruction 설정 (영어)")
    print("""
  현재 사용 중인 Instruction (mobile_vla_action_dataset.py Line 142-151):
  
  LEFT scenarios:
    - "Navigate around the obstacle on the left side and reach the cup"
    
  RIGHT scenarios:  
    - "Navigate around the obstacle on the right side and reach the cup"
    
  ※ 한국어가 아닌 영어로 설정됨 (VLA 표준: OpenVLA, RT-2와 동일)
""")
    
    # 모델 로드
    print_section("모델 구조 분석")
    
    ckpts = glob.glob("runs/**/last.ckpt", recursive=True)
    if ckpts:
        checkpoint_path = max(ckpts, key=os.path.getmtime)
        print(f"\n[Checkpoint] {checkpoint_path}")
        model = MobileVLATrainer.load_from_checkpoint(
            checkpoint_path, configs=configs, strict=False, map_location='cpu'
        )
    else:
        print("\n[새 모델 초기화]")
        model = MobileVLATrainer(configs)
    
    # 모델 구조 분석
    backbone = model.model  # BaseRoboVLM (RoboKosmos)
    
    # 구성 요소별 파라미터 분석
    print_section("Frozen/Trainable 상세 분석", "-")
    
    components = {}
    
    for name, param in model.named_parameters():
        # 구성 요소 분류
        if 'vision_model' in name or 'vision_tower' in name or 'image_encoder' in name:
            comp = "Vision Encoder"
        elif 'text_model' in name or 'text_tower' in name or 'language_model' in name:
            comp = "Text Encoder (LLM)"
        elif 'embed_tokens' in name or 'word_embedding' in name or 'embedding' in name.lower():
            comp = "Word Embedding"
        elif 'act_head' in name or 'action_head' in name:
            comp = "Action Head (LSTM)"
        elif 'velocities' in name:
            comp = "Action Head (MLP)"
        elif 'rnn' in name:
            comp = "Action Head (RNN)"
        elif 'action_token' in name:
            comp = "Action Token"
        else:
            comp = "Other (VLM)"
        
        if comp not in components:
            components[comp] = {"total": 0, "trainable": 0, "frozen": 0, "params": []}
        
        components[comp]["total"] += param.numel()
        if param.requires_grad:
            components[comp]["trainable"] += param.numel()
            components[comp]["params"].append(f"  ✅ {name}: {param.shape}")
        else:
            components[comp]["frozen"] += param.numel()
            # Frozen 파라미터는 처음 몇 개만 표시
            if len([p for p in components[comp]["params"] if "❄️" in p]) < 3:
                components[comp]["params"].append(f"  ❄️ {name}: {param.shape}")
    
    # 결과 출력
    print(f"\n{'구성 요소':<25} {'Total':>12} {'Trainable':>12} {'Frozen':>12} {'상태':>10}")
    print("-" * 75)
    
    for comp, data in sorted(components.items()):
        total = data["total"]
        trainable = data["trainable"]
        frozen = data["frozen"]
        
        if trainable > frozen:
            status = "✅ 학습"
        elif frozen > 0 and trainable == 0:
            status = "❄️ 고정"
        else:
            status = "⚠️ 혼합"
        
        print(f"{comp:<25} {total:>12,} {trainable:>12,} {frozen:>12,} {status:>10}")
    
    # 상세 파라미터 목록
    print_section("Trainable 파라미터 상세 목록", "-")
    
    for comp, data in sorted(components.items()):
        if data["trainable"] > 0:
            print(f"\n[{comp}]")
            for p in data["params"]:
                if "✅" in p:
                    print(p)
    
    # Frozen 구성 요소 요약
    print_section("Frozen 구성 요소 (학습 안 됨)", "-")
    
    for comp, data in sorted(components.items()):
        if data["frozen"] > 0 and data["trainable"] == 0:
            print(f"\n[{comp}] - 완전 고정")
            count = 0
            for p in data["params"]:
                if "❄️" in p:
                    print(p)
                    count += 1
                if count >= 3:
                    remaining = len([x for x in data["params"] if "❄️" in x]) - 3
                    if remaining > 0:
                        print(f"  ... 외 {remaining}개 더")
                    break
    
    # 최종 요약
    print_section("최종 요약")
    
    total_params = sum(d["total"] for d in components.values())
    trainable_params = sum(d["trainable"] for d in components.values())
    frozen_params = sum(d["frozen"] for d in components.values())
    
    print(f"""
  총 파라미터:         {total_params:>15,}
  Trainable 파라미터:  {trainable_params:>15,} ({100*trainable_params/total_params:.2f}%)
  Frozen 파라미터:     {frozen_params:>15,} ({100*frozen_params/total_params:.2f}%)
  
  [Frozen 구성 요소]
  - Vision Encoder (이미지 처리)
  - Text Encoder / LLM (언어 처리)
  - Word Embedding (토큰 임베딩)
  
  [Trainable 구성 요소]
  - Action Head (LSTM + MLP)
  - Action Token
  
  → VLM이 Frozen이므로 instruction embedding이 고정됨
  → Action Head만 학습되어 instruction을 구분하지 못함
""")
    
    print_section("권장 해결책")
    print("""
  [옵션 1] LoRA Fine-tuning (추천)
    - freeze_backbone: false
    - lora_enable: true
    - lora_target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
    - VLM의 attention layer만 학습하여 instruction grounding 개선
    
  [옵션 2] Text Embedding 학습
    - freeze_backbone: true
    - train_text_embedding: true
    - Word Embedding만 학습 (메모리 효율적)
    
  [옵션 3] Decoder Layer 학습
    - freeze_backbone: false
    - train_decoder_layers: 4  (마지막 4개 layer만 학습)
""")


if __name__ == "__main__":
    main()
