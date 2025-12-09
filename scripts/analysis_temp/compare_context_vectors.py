#!/usr/bin/env python3
"""
Context Vector 비교: Kosmos-2 vs RoboVLMs
목적: VLM pretrain 차이가 context vector에 미치는 영향 분석
"""

import torch
import numpy as np
import h5py
from pathlib import Path
import sys
from PIL import Image
import torchvision.transforms as T
import json

sys.path.insert(0, "RoboVLMs_upstream")

def load_checkpoint_and_extract_vlm(ckpt_path, device='cuda'):
    """
    Checkpoint에서 VLM만 추출
    """
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    
    # State dict에서 VLM 부분만 추출
    vlm_state_dict = {}
    for key, value in checkpoint.items():
        if 'state_dict' in checkpoint:
            # Lightning checkpoint
            for k, v in checkpoint['state_dict'].items():
                if 'model.model' in k and 'act_head' not in k:
                    # Remove 'model.model.' prefix
                    new_key = k.replace('model.model.', '')
                    vlm_state_dict[new_key] = v
        else:
            # Direct checkpoint
            if 'act_head' not in key and 'action' not in key:
                vlm_state_dict[key] = value
    
    print(f"  Extracted {len(vlm_state_dict)} VLM parameters")
    return vlm_state_dict

def compare_context_vectors():
    """
    Context vector 비교 분석
    """
    print("="*70)
    print("Context Vector 비교: Kosmos-2 vs RoboVLMs")
    print("="*70)
    
    # 1. 체크포인트 경로
    kosmos2_ckpt = "RoboVLMs_upstream/runs/mobile_vla_lora_20251203/kosmos/mobile_vla_finetune/2025-12-03/mobile_vla_lora_20251203/epoch_epoch=09-val_loss=val_loss=0.013.ckpt"
    robovlms_ckpt = "checkpoints/RoboVLMs/checkpoints/kosmos_ph_oxe-pretrain.pt"
    
    print("\n[1] 체크포인트 확인")
    print("-"*70)
    
    # Kosmos-2 확인
    if Path(kosmos2_ckpt).exists():
        size_k = Path(kosmos2_ckpt).stat().st_size / (1024**3)
        print(f"  ✅ Kosmos-2 (Mobile-VLA): {size_k:.2f} GB")
    else:
        print(f"  ❌ Kosmos-2 없음: {kosmos2_ckpt}")
        return
    
    # RoboVLMs 확인  
    if Path(robovlms_ckpt).exists():
        size_r = Path(robovlms_ckpt).stat().st_size / (1024**3)
        print(f"  ✅ RoboVLMs: {size_r:.2f} GB")
    else:
        print(f"  ❌ RoboVLMs 없음: {robovlms_ckpt}")
        print("  → HuggingFace에서 다운로드 필요")
        return
    
    # 2. 모델 로드
    print("\n[2] VLM 로드 및 비교")
    print("-"*70)
    
    try:
        from robovlms.train.mobile_vla_trainer import MobileVLATrainer
        
        # Kosmos-2 (Mobile-VLA)
        print("  Loading Kosmos-2 (Mobile-VLA trained)...")
        model_k = MobileVLATrainer.load_from_checkpoint(kosmos2_ckpt)
        model_k.eval()
        model_k.cuda()
        print("  ✅ Kosmos-2 loaded")
        
        # RoboVLMs는 직접 checkpoint 분석
        print("  Analyzing RoboVLMs checkpoint...")
        vlm_state_dict = load_checkpoint_and_extract_vlm(robovlms_ckpt)
        print("  ✅ RoboVLMs analyzed")
        
    except Exception as e:
        print(f"  ❌ 모델 로드 실패: {e}")
        return
    
    # 3. 테스트 이미지로 context 추출
    print("\n[3] Context Vector 추출")
    print("-"*70)
    
    # 테스트 이미지 로드
    h5_file = "ROS_action/mobile_vla_dataset/episode_20251204_113519_1box_hori_left_core_medium.h5"
    
    if not Path(h5_file).exists():
        # 다른 파일 찾기
        import glob
        h5_files = glob.glob("ROS_action/mobile_vla_dataset/episode_*.h5")
        if h5_files:
            h5_file = h5_files[0]
            print(f"  Using: {Path(h5_file).name}")
        else:
            print("  ❌ H5 파일 없음")
            return
    
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])
    
    # 이미지 로드
    with h5py.File(h5_file, 'r') as f:
        images = []
        for t in range(min(8, len(f['images']))):
            img_array = f['images'][t]
            img = Image.fromarray(img_array.astype(np.uint8))
            img_tensor = transform(img)
            images.append(img_tensor)
        
        images_tensor = torch.stack(images).unsqueeze(0).cuda()
        print(f"  이미지 shape: {images_tensor.shape}")
    
    # Kosmos-2 context 추출
    with torch.no_grad():
        context_k = model_k.model.encode_images(images_tensor)
    
    print(f"\n  Kosmos-2 Context:")
    print(f"    Shape: {context_k.shape}")
    print(f"    Mean: {context_k.mean().item():.4f}")
    print(f"    Std: {context_k.std().item():.4f}")
    print(f"    Min: {context_k.min().item():.4f}")
    print(f"    Max: {context_k.max().item():.4f}")
    
    # 4. 분석
    print("\n[4] 분석 결과")
    print("="*70)
    
    print("\n✅ Kosmos-2 (Mobile-VLA trained):")
    print("  - Pretrain: 일반 이미지 (COCO, Flickr)")
    print("  - Fine-tuned: Mobile navigation (250 left)")
    print("  - Context: Image → 2048D vector")
    
    print("\n📊 RoboVLMs:")
    print("  - Pretrain: Robot manipulation (OXE)")
    print("  - Checkpoint 분석 완료")
    print(f"  - Parameters: {len(vlm_state_dict)}")
    
    print("\n🎯 결론:")
    print("  1. RoboVLMs checkpoint 다운로드 성공")
    print("  2. Kosmos-2 context vector 추출 성공")
    print("  3. VLM pretrain 차이 확인 가능")
    print("  4. 다음: RoboVLMs로 학습하여 context 비교")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    compare_context_vectors()
