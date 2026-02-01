#!/usr/bin/env python3
"""
의미 벡터 비교: Frozen VLM vs LoRA VLM

목적:
1. Case 1 (LoRA + Action Head): VLM이 태스크에 적응한 의미 벡터
2. Case 2 (Frozen + Action Head): VLM 원본 의미 벡터

비교 메트릭:
- Cosine Similarity
- L2 Distance
- CKA (Centered Kernel Alignment)
"""

import torch
import torch.nn.functional as F
import numpy as np
import h5py
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
import sys
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent / "RoboVLMs_upstream"))

from robovlms.train.mobile_vla_trainer import MobileVLATrainer


def compute_cka(X, Y):
    """
    Centered Kernel Alignment (CKA) - 표현 유사도 측정
    
    CKA는 두 표현 공간의 구조적 유사성을 측정합니다.
    1.0에 가까울수록 유사한 표현
    """
    def centering(K):
        n = K.shape[0]
        unit = np.ones([n, n])
        I = np.eye(n)
        H = I - unit / n
        return np.dot(np.dot(H, K), H)
    
    def linear_HSIC(X, Y):
        L_X = np.dot(X, X.T)
        L_Y = np.dot(Y, Y.T)
        return np.sum(centering(L_X) * centering(L_Y))
    
    hsic_xy = linear_HSIC(X, Y)
    hsic_xx = linear_HSIC(X, X)
    hsic_yy = linear_HSIC(Y, Y)
    
    return hsic_xy / (np.sqrt(hsic_xx) * np.sqrt(hsic_yy) + 1e-10)


def extract_semantic_vectors(model, h5_files, transform, device, max_samples=50):
    """
    모델에서 의미 벡터 추출
    
    Returns:
        vectors: (N, D) 형태의 numpy array
        labels: Left/Right 라벨
    """
    vectors = []
    labels = []
    
    with torch.no_grad():
        for i, h5_file in enumerate(h5_files[:max_samples]):
            with h5py.File(h5_file, 'r') as f:
                # 이미지 로드
                images = []
                for t in range(min(8, len(f['images']))):
                    img = Image.fromarray(f['images'][t].astype(np.uint8))
                    images.append(transform(img))
                
                while len(images) < 8:
                    images.append(torch.zeros(3, 224, 224))
                
                images_tensor = torch.stack(images).unsqueeze(0).to(device)
                
                # 의미 벡터 추출 (encode_images)
                context = model.model.encode_images(images_tensor)
                
                # Flatten: (1, 8, 64, 2048) -> (1, D)
                vector = context.view(1, -1).cpu().numpy()
                vectors.append(vector)
                
                # 라벨
                label = 'left' if 'left' in str(h5_file) else 'right'
                labels.append(label)
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{min(len(h5_files), max_samples)}")
    
    return np.vstack(vectors), labels


def compare_vectors(vectors_frozen, vectors_lora):
    """
    두 모델의 의미 벡터 비교
    """
    results = {}
    
    # 1. Cosine Similarity (샘플별)
    cosine_sims = []
    for v1, v2 in zip(vectors_frozen, vectors_lora):
        sim = F.cosine_similarity(
            torch.tensor(v1).unsqueeze(0),
            torch.tensor(v2).unsqueeze(0)
        ).item()
        cosine_sims.append(sim)
    
    results['cosine_mean'] = np.mean(cosine_sims)
    results['cosine_std'] = np.std(cosine_sims)
    
    # 2. L2 Distance
    l2_dists = np.linalg.norm(vectors_frozen - vectors_lora, axis=1)
    results['l2_mean'] = np.mean(l2_dists)
    results['l2_std'] = np.std(l2_dists)
    
    # 3. CKA (Centered Kernel Alignment)
    results['cka'] = compute_cka(vectors_frozen, vectors_lora)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Frozen vs LoRA 의미 벡터 비교")
    parser.add_argument("--frozen_ckpt", type=str, required=True, help="Frozen 모델 체크포인트")
    parser.add_argument("--lora_ckpt", type=str, default=None, help="LoRA 모델 체크포인트 (없으면 비교 스킵)")
    parser.add_argument("--data_dir", type=str, default="ROS_action/mobile_vla_dataset")
    parser.add_argument("--max_samples", type=int, default=50)
    args = parser.parse_args()
    
    print("="*70)
    print("의미 벡터 비교: Frozen VLM vs LoRA VLM")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
    
    # 데이터 로드
    h5_files = sorted(list(Path(args.data_dir).glob("episode_*.h5")))
    print(f"\n총 에피소드: {len(h5_files)}")
    
    # Frozen 모델 로드
    print(f"\n[1/3] Frozen 모델 로드: {args.frozen_ckpt}")
    model_frozen = MobileVLATrainer.load_from_checkpoint(args.frozen_ckpt, map_location='cpu')
    model_frozen.eval()
    model_frozen.to(device)
    
    # Frozen 벡터 추출
    print("\n[2/3] Frozen 의미 벡터 추출...")
    vectors_frozen, labels = extract_semantic_vectors(
        model_frozen, h5_files, transform, device, args.max_samples
    )
    print(f"  Shape: {vectors_frozen.shape}")
    
    # LoRA 모델 (있으면)
    if args.lora_ckpt and Path(args.lora_ckpt).exists():
        print(f"\n[3/3] LoRA 모델 로드: {args.lora_ckpt}")
        model_lora = MobileVLATrainer.load_from_checkpoint(args.lora_ckpt, map_location='cpu')
        model_lora.eval()
        model_lora.to(device)
        
        print("\n  LoRA 의미 벡터 추출...")
        vectors_lora, _ = extract_semantic_vectors(
            model_lora, h5_files, transform, device, args.max_samples
        )
        
        # 비교
        print("\n" + "="*70)
        print("비교 결과")
        print("="*70)
        
        results = compare_vectors(vectors_frozen, vectors_lora)
        
        print(f"\n📊 Cosine Similarity: {results['cosine_mean']:.4f} ± {results['cosine_std']:.4f}")
        print(f"📊 L2 Distance: {results['l2_mean']:.4f} ± {results['l2_std']:.4f}")
        print(f"📊 CKA: {results['cka']:.4f}")
        
        # 해석
        print("\n📝 해석:")
        if results['cosine_mean'] > 0.9:
            print("  - Cosine > 0.9: 매우 유사한 방향 → LoRA가 벡터 방향 크게 변경 안 함")
        elif results['cosine_mean'] > 0.7:
            print("  - 0.7 < Cosine < 0.9: 중간 정도 유사 → LoRA가 일부 적응")
        else:
            print("  - Cosine < 0.7: 상당히 다름 → LoRA가 표현 크게 변경")
        
        if results['cka'] > 0.8:
            print("  - CKA > 0.8: 구조적으로 유사한 표현 공간")
        else:
            print("  - CKA < 0.8: 구조적으로 다른 표현 공간")
    else:
        print("\n[3/3] LoRA 체크포인트 없음 - 비교 스킵")
        print("  → Case 1 (LoRA) 학습 후 다시 실행하세요")
    
    # Left vs Right 분석 (Frozen만)
    print("\n" + "="*70)
    print("Frozen 모델 내 Left vs Right 비교")
    print("="*70)
    
    left_mask = np.array([l == 'left' for l in labels])
    right_mask = ~left_mask
    
    vectors_left = vectors_frozen[left_mask]
    vectors_right = vectors_frozen[right_mask]
    
    # 평균 벡터 간 유사도
    mean_left = vectors_left.mean(axis=0)
    mean_right = vectors_right.mean(axis=0)
    
    cosine_lr = F.cosine_similarity(
        torch.tensor(mean_left).unsqueeze(0),
        torch.tensor(mean_right).unsqueeze(0)
    ).item()
    
    l2_lr = np.linalg.norm(mean_left - mean_right)
    
    print(f"\n📊 Left vs Right (Frozen 모델):")
    print(f"  Cosine Similarity: {cosine_lr:.4f}")
    print(f"  L2 Distance: {l2_lr:.4f}")
    
    if cosine_lr > 0.95:
        print("  → 의미 벡터에서 Left/Right 구분 약함 (이미지 기반)")
    else:
        print("  → 의미 벡터에서 Left/Right 어느 정도 구분됨")
    
    print("\n완료!")


if __name__ == "__main__":
    main()
