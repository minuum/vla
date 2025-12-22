#!/usr/bin/env python3
"""
Extract Context Vectors: Fine-Tuned vs Pre-trained
==================================================

목적: LoRA Fine-Tuning이 latent space를 어떻게 변화시켰는지 분석

비교:
1. Pre-trained Kosmos-2 (학습 전)
2. Fine-tuned Model (Case 5, Epoch 4)

Output:
  - pretrained_left_contexts.npy
  - pretrained_right_contexts.npy  
  - finetuned_left_contexts.npy
  - finetuned_right_contexts.npy
"""

import torch
import numpy as np
import h5py
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import json
import sys

# Add RoboVLMs to path
sys.path.append(str(Path.cwd() / "RoboVLMs_upstream"))

# Configuration
FINETUNED_CHECKPOINT = "runs/mobile_vla_no_chunk_20251209/kosmos/mobile_vla_finetune/2025-12-09/mobile_vla_no_chunk_20251209/epoch_epoch=04-val_loss=val_loss=0.001.ckpt"
DATASET_PATH = Path("ROS_action/mobile_vla_dataset")
OUTPUT_DIR = Path("docs/latent_space_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SIZE = 8
MAX_EPISODES = 50  # 각 방향당

# Image transform (Kosmos-2 style)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )
])

print("="*80)
print("Context Vector Extraction: Fine-Tuned vs Pre-trained")
print("="*80)
print()

# ============================================================================
# Helper Functions
# ============================================================================

def load_episodes(pattern, max_episodes=50):
    """Load H5 episode files"""
    files = sorted(DATASET_PATH.glob(pattern))
    print(f"  Found: {len(files)} files")
    print(f"  Using: {min(max_episodes, len(files))} files")
    return files[:max_episodes]

def process_episode(h5_file):
    """Extract frames from H5 file"""
    try:
        with h5py.File(h5_file, 'r') as f:
            images = f['images']
            
            frames = []
            for i in range(min(WINDOW_SIZE, len(images))):
                img = Image.fromarray(images[i].astype(np.uint8))
                frames.append(transform(img))
            
            # Pad if needed
            while len(frames) < WINDOW_SIZE:
                frames.append(frames[-1])
            
            # Stack: [8, 3, 224, 224]
            return torch.stack(frames)
            
    except Exception as e:
        print(f"  ⚠️  Error: {h5_file.name}: {e}")
        return None

# ============================================================================
# Step 1: Load Episodes
# ============================================================================

print("[1/5] Loading episodes...")
left_files = load_episodes("*left*.h5", MAX_EPISODES)
right_files = load_episodes("*right*.h5", MAX_EPISODES)
print()

# ============================================================================
# Step 2: Extract Pre-trained Contexts
# ============================================================================

print("[2/5] Extracting Pre-trained contexts...")
print("  ⚠️  NOTE: Requires Kosmos-2 pre-trained model")
print("  Current: Using placeholder (random)")
print()

pretrained_left = []
pretrained_right = []

for h5_file in tqdm(left_files, desc="Pre-Left"):
    frames = process_episode(h5_file)
    if frames is not None:
        # TODO: Replace with actual model inference
        # context = pretrained_model.encode_images(frames)
        # For now: placeholder
        context = np.random.randn(WINDOW_SIZE, 64, 2048)
        pretrained_left.append(context)

for h5_file in tqdm(right_files, desc="Pre-Right"):
    frames = process_episode(h5_file)
    if frames is not None:
        context = np.random.randn(WINDOW_SIZE, 64, 2048)
        pretrained_right.append(context)

pretrained_left = np.array(pretrained_left)
pretrained_right = np.array(pretrained_right)

print(f"  Pre-Left shape: {pretrained_left.shape}")
print(f"  Pre-Right shape: {pretrained_right.shape}")
print()

# ============================================================================
# Step 3: Extract Fine-tuned Contexts
# ============================================================================

print("[3/5] Extracting Fine-tuned contexts...")
print(f"  Checkpoint: {FINETUNED_CHECKPOINT}")
print("  ⚠️  NOTE: Requires loading trained model")
print("  Current: Using placeholder (random)")
print()

finetuned_left = []
finetuned_right = []

# TODO: Load fine-tuned checkpoint
# checkpoint = torch.load(FINETUNED_CHECKPOINT)
# model.load_state_dict(checkpoint['state_dict'])

for h5_file in tqdm(left_files, desc="FT-Left"):
    frames = process_episode(h5_file)
    if frames is not None:
        # TODO: Replace with actual model inference
        # context = finetuned_model.encode_images(frames)
        context = np.random.randn(WINDOW_SIZE, 64, 2048)
        finetuned_left.append(context)

for h5_file in tqdm(right_files, desc="FT-Right"):
    frames = process_episode(h5_file)
    if frames is not None:
        context = np.random.randn(WINDOW_SIZE, 64, 2048)
        finetuned_right.append(context)

finetuned_left = np.array(finetuned_left)
finetuned_right = np.array(finetuned_right)

print(f"  FT-Left shape: {finetuned_left.shape}")
print(f"  FT-Right shape: {finetuned_right.shape}")
print()

# ============================================================================
# Step 4: Save
# ============================================================================

print("[4/5] Saving context vectors...")

files_to_save = {
    'pretrained_left_contexts.npy': pretrained_left,
    'pretrained_right_contexts.npy': pretrained_right,
    'finetuned_left_contexts.npy': finetuned_left,
    'finetuned_right_contexts.npy': finetuned_right,
}

for filename, data in files_to_save.items():
    path = OUTPUT_DIR / filename
    np.save(path, data)
    print(f"  ✅ {filename}: {data.shape}")

# Metadata
metadata = {
    'pretrained': {
        'left_shape': list(pretrained_left.shape),
        'right_shape': list(pretrained_right.shape),
        'source': 'Kosmos-2 pre-trained (placeholder)'
    },
    'finetuned': {
        'left_shape': list(finetuned_left.shape),
        'right_shape': list(finetuned_right.shape),
        'checkpoint': str(FINETUNED_CHECKPOINT),
        'source': 'Case 5 Epoch 4 (placeholder)'
    },
    'parameters': {
        'window_size': WINDOW_SIZE,
        'max_episodes': MAX_EPISODES,
        'feature_dim': 2048,
        'n_tokens': 64
    }
}

with open(OUTPUT_DIR / "metadata.json", 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"  ✅ metadata.json")
print()

# ============================================================================
# Step 5: Quick Comparison
# ============================================================================

print("[5/5] Quick similarity check...")

from scipy.spatial.distance import cosine

def avg_cosine_similarity(arr1, arr2):
    """Average pairwise cosine similarity"""
    arr1_flat = arr1.reshape(arr1.shape[0], -1)
    arr2_flat = arr2.reshape(arr2.shape[0], -1)
    
    sims = []
    for i in range(min(10, len(arr1_flat))):
        for j in range(min(10, len(arr2_flat))):
            sim = 1 - cosine(arr1_flat[i], arr2_flat[j])
            sims.append(sim)
    
    return np.mean(sims)

print("  Pre-trained:")
print(f"    Left-Left: {avg_cosine_similarity(pretrained_left, pretrained_left):.4f}")
print(f"    Right-Right: {avg_cosine_similarity(pretrained_right, pretrained_right):.4f}")
print(f"    Left-Right: {avg_cosine_similarity(pretrained_left, pretrained_right):.4f}")

print("  Fine-tuned:")
print(f"    Left-Left: {avg_cosine_similarity(finetuned_left, finetuned_left):.4f}")
print(f"    Right-Right: {avg_cosine_similarity(finetuned_right, finetuned_right):.4f}")
print(f"    Left-Right: {avg_cosine_similarity(finetuned_left, finetuned_right):.4f}")

print()
print("="*80)
print("✅ Extraction complete!")
print("="*80)
print()
print("⚠️  WARNING: Using PLACEHOLDER context vectors!")
print("   Real implementation requires:")
print("   1. Loading Kosmos-2 pre-trained model")
print("   2. Loading Case 5 fine-tuned checkpoint")
print("   3. Proper forward pass to extract hidden states")
print()
print("Next steps:")
print("  1. Implement actual model loading")
print("  2. Run: python3 docs/latent_space_analysis/compare_ft_noFT.py")
print("  3. Run: python3 docs/latent_space_analysis/visualize_comparison.py")
