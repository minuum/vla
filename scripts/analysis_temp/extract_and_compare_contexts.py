#!/usr/bin/env python3
"""
실제 Context Vector 추출 및 비교 스크립트

목적:
1. Mobile-VLA (trained Kosmos-2)의 context vector 추출
2. RoboVLMs (pretrained)의 context vector 추출  
3. 두 context 비교 (mean, std, distribution)
4. 시각화 및 저장
"""

import torch
import numpy as np
import h5py
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
import json
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, "RoboVLMs_upstream")


def load_sample_images(num_episodes=10):
    """
    샘플 이미지 로드
    
    Returns:
        tensor: (num_episodes, 8, 3, 224, 224)
    """
    print(f"\n[1] 샘플 데이터 로드 ({num_episodes} episodes)")
    
    h5_files = sorted(list(Path("ROS_action/mobile_vla_dataset").glob("episode_*.h5")))
    
    # 균형있게 샘플링 (left + right)
    left_files = [f for f in h5_files if 'left' in str(f)][:num_episodes//2]
    right_files = [f for f in h5_files if 'right' in str(f)][:num_episodes//2]
    selected_files = left_files + right_files
    
    print(f"  Left: {len(left_files)}, Right: {len(right_files)}")
    
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])
    
    sample_images = []
    for h5_file in selected_files:
        try:
            with h5py.File(h5_file, 'r') as f:
                # 8 프레임 로드
                frames = []
                for i in range(min(8, len(f['images']))):
                    img = Image.fromarray(f['images'][i].astype(np.uint8))
                    frames.append(transform(img))
                
                # Padding if needed
                while len(frames) < 8:
                    frames.append(torch.zeros(3, 224, 224))
                
                sample_images.append(torch.stack(frames))
        except Exception as e:
            print(f"  ⚠️  {h5_file.name}: {e}")
            continue
    
    if len(sample_images) == 0:
        raise ValueError("샘플 이미지 로드 실패")
    
    images_batch = torch.stack(sample_images)
    print(f"  ✅ Shape: {images_batch.shape}")
    
    return images_batch


def extract_mobile_vla_context(images):
    """
    Mobile-VLA (trained) context vector 추출
    
    Args:
        images: (N, 8, 3, 224, 224)
    
    Returns:
        context: (N, 8, 64, 2048)
    """
    print("\n[2] Mobile-VLA Context 추출")
    
    from robovlms.train.mobile_vla_trainer import MobileVLATrainer
    
    ckpt_path = "RoboVLMs_upstream/runs/mobile_vla_lora_20251203/kosmos/mobile_vla_finetune/2025-12-03/mobile_vla_lora_20251203/epoch_epoch=09-val_loss=val_loss=0.013.ckpt"
    
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"Checkpoint 없음: {ckpt_path}")
    
    print(f"  로딩: {Path(ckpt_path).name}")
    model = MobileVLATrainer.load_from_checkpoint(ckpt_path, map_location='cpu')
    model.eval()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    images = images.to(device)
    
    print(f"  Device: {device}")
    
    with torch.no_grad():
        context = model.model.encode_images(images)
    
    print(f"  ✅ Context shape: {context.shape}")
    
    return context.cpu()


def extract_robovlms_context(images):
    """
    RoboVLMs (pretrained) context vector 추출
    
    ⚠️  현재 미구현 - checkpoint 구조 분석 필요
    """
    print("\n[3] RoboVLMs Context 추출")
    print("  ⚠️  TODO: RoboVLMs checkpoint 구조 분석 필요")
    print("  - Checkpoint: .vlms/RoboVLMs/checkpoints/kosmos_ph_oxe-pretrain.pt")
    print("  - 구조가 Mobile-VLA와 다를 수 있음")
    
    return None


def compute_statistics(context, name="Context"):
    """
    Context vector 통계 계산
    """
    stats = {
        'name': name,
        'shape': list(context.shape),
        'mean': float(context.mean()),
        'std': float(context.std()),
        'min': float(context.min()),
        'max': float(context.max()),
        'norm': float(torch.norm(context)),
    }
    
    # 출력
    print(f"\n  {name} 통계:")
    print(f"    Shape: {stats['shape']}")
    print(f"    Mean:  {stats['mean']:.4f}")
    print(f"    Std:   {stats['std']:.4f}")
    print(f"    Min:   {stats['min']:.4f}")
    print(f"    Max:   {stats['max']:.4f}")
    print(f"    Norm:  {stats['norm']:.2f}")
    
    return stats


def visualize_context(context, name="Context", save_path="context_viz.png"):
    """
    Context vector 시각화
    """
    print(f"\n[Viz] {name} 시각화")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Distribution histogram
    axes[0, 0].hist(context.flatten().numpy(), bins=100, alpha=0.7)
    axes[0, 0].set_title(f'{name} Distribution')
    axes[0, 0].set_xlabel('Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].axvline(context.mean(), color='r', linestyle='--', label=f'Mean={context.mean():.3f}')
    axes[0, 0].legend()
    
    # 2. Heatmap (첫 번째 샘플, 첫 번째 프레임)
    if len(context.shape) >= 4:
        # (N, frames, tokens, features)
        sample = context[0, 0].numpy()  # (64, 2048)
    else:
        sample = context[0].numpy()
    
    im = axes[0, 1].imshow(sample, aspect='auto', cmap='viridis')
    axes[0, 1].set_title(f'{name} Heatmap (Sample 0, Frame 0)')
    axes[0, 1].set_xlabel('Features (2048)')
    axes[0, 1].set_ylabel('Tokens (64)')
    plt.colorbar(im, ax=axes[0, 1])
    
    # 3. Mean across tokens
    mean_features = sample.mean(axis=0)  # (2048,)
    axes[1, 0].plot(mean_features)
    axes[1, 0].set_title(f'{name} Mean Features (across tokens)')
    axes[1, 0].set_xlabel('Feature Index')
    axes[1, 0].set_ylabel('Mean Value')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Std across tokens
    std_features = sample.std(axis=0)  # (2048,)
    axes[1, 1].plot(std_features)
    axes[1, 1].set_title(f'{name} Std Features (across tokens)')
    axes[1, 1].set_xlabel('Feature Index')
    axes[1, 1].set_ylabel('Std Value')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  ✅ 저장: {save_path}")
    
    plt.close()


def main():
    print("="*70)
    print(" Context Vector 실제 추출 및 비교")
    print("="*70)
    
    # 1. 샘플 이미지 로드
    images = load_sample_images(num_episodes=10)
    
    # 2. Mobile-VLA context 추출
    mobile_context = extract_mobile_vla_context(images)
    mobile_stats = compute_statistics(mobile_context, "Mobile-VLA")
    
    # 3. RoboVLMs context 추출 (TODO)
    robovlms_context = extract_robovlms_context(images)
    
    # 4. 시각화
    print("\n[4] 시각화")
    visualize_context(mobile_context, "Mobile-VLA", "mobile_vla_context.png")
    
    # 5. 결과 저장
    print("\n[5] 결과 저장")
    results = {
        'mobile_vla': mobile_stats,
        'note': 'RoboVLMs context extraction TODO'
    }
    
    with open('context_comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"  ✅ context_comparison_results.json")
    
    print("\n" + "="*70)
    print("완료!")
    print("="*70)
    print("\n생성된 파일:")
    print("  - context_comparison_results.json")
    print("  - mobile_vla_context.png")
    print("\nTODO:")
    print("  - RoboVLMs checkpoint 구조 분석")
    print("  - RoboVLMs context 추출 구현")
    print("  - 두 context 비교 (cosine similarity, distribution)")


if __name__ == "__main__":
    main()
