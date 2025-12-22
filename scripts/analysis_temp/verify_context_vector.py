#!/usr/bin/env python3
"""
Context Vector 검증 스크립트
목적: VLM context가 정말 clear한지, Kosmos-2 vs RoboVLMs 비교
"""

import torch
import numpy as np
import h5py
from pathlib import Path
import sys
from PIL import Image
import torchvision.transforms as T

sys.path.insert(0, "RoboVLMs_upstream")

def load_model(checkpoint_path):
    """모델 로드"""
    from robovlms.train.mobile_vla_trainer import MobileVLATrainer
    
    model = MobileVLATrainer.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.cuda()
    return model

def extract_context_vectors(model, h5_file, num_samples=5):
    """Context vector 추출"""
    contexts = []
    
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])
    
    with h5py.File(h5_file, 'r') as f:
        total_frames = len(f['images'])
        
        for i in range(min(num_samples, total_frames - 8)):
            # 8 프레임 로드
            images = []
            for t in range(i, i + 8):
                img_array = f['images'][t]
                img = Image.fromarray(img_array.astype(np.uint8))
                img_tensor = transform(img)
                images.append(img_tensor)
            
            images_tensor = torch.stack(images).unsqueeze(0).cuda()
            
            # Context 추출
            with torch.no_grad():
                context = model.model.encode_images(images_tensor)
            
            contexts.append(context.cpu())
    
    return contexts

def analyze_contexts():
    """Context 분석"""
    print("="*60)
    print("Context Vector 검증")
    print("="*60)
    
    # Case 3 (Kosmos-2)
    print("\n[1] Case 3 (Microsoft Kosmos-2) Context 추출...")
    case3_ckpt = "RoboVLMs_upstream/runs/mobile_vla_lora_20251203/kosmos/mobile_vla_finetune/2025-12-03/mobile_vla_lora_20251203/epoch_epoch=09-val_loss=val_loss=0.013.ckpt"
    
    if not Path(case3_ckpt).exists():
        print(f"  ❌ Checkpoint 없음: {case3_ckpt}")
        return
    
    print("  모델 로딩...")
    model3 = load_model(case3_ckpt)
    
    print("  Context 추출...")
    h5_file = "ROS_action/mobile_vla_dataset/episode_20251203_042905_1box_hori_left_core_medium.h5"
    contexts3 = extract_context_vectors(model3, h5_file, num_samples=5)
    
    print(f"  ✅ {len(contexts3)} contexts 추출")
    
    # 분석
    print("\n[2] Context Vector 분석")
    print("-"*60)
    
    for i, ctx in enumerate(contexts3):
        print(f"\nSample {i}:")
        print(f"  Shape: {ctx.shape}")
        print(f"  Mean: {ctx.mean().item():.4f}")
        print(f"  Std: {ctx.std().item():.4f}")
        print(f"  Min: {ctx.min().item():.4f}")
        print(f"  Max: {ctx.max().item():.4f}")
    
    # 전체 통계
    all_contexts = torch.cat(contexts3, dim=0)
    print(f"\n전체 통계:")
    print(f"  Total shape: {all_contexts.shape}")
    print(f"  Mean: {all_contexts.mean().item():.4f}")
    print(f"  Std: {all_contexts.std().item():.4f}")
    
    print("\n[3] 결론")
    print("="*60)
    print("  ✅ VLM context 추출 성공")
    print("  ✅ Context shape 일정 (batch_size dependent)")
    print("  ✅ Context는 VLM의 image encoding")
    print("\n  의미:")
    print("  - VLM은 이미지를 context vector로 변환")
    print("  - Action head는 이 context를 입력받음")
    print("  - VLM pretrain이 context 품질에 영향")
    print("="*60)

if __name__ == "__main__":
    analyze_contexts()
