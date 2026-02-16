#!/usr/bin/env python3
"""
Extract Left vs Right Direction Context Vectors
수요일 미팅 준비: Latent Space 분석

목적: Frozen VLM의 latent space에서 Left와 Right direction의 
     의미 벡터(context vector)가 어떻게 다른지 비교

Input: Case 5 checkpoint (best model, epoch 4)
Output: 
  - left_contexts.npy: [N, 8, 64, 2048]
  - right_contexts.npy: [N, 8, 64, 2048]
"""

import torch
import numpy as np
import h5py
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import json

# Setup
CHECKPOINT_PATH = "runs/mobile_vla_no_chunk_20251209/kosmos/mobile_vla_finetune/2025-12-09/mobile_vla_no_chunk_20251209/epoch_epoch=04-val_loss=val_loss=0.001.ckpt"
DATASET_PATH = Path("ROS_action/mobile_vla_dataset")
OUTPUT_DIR = Path("docs/latent_space_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Parameters
WINDOW_SIZE = 8
MAX_EPISODES_PER_DIR = 50  # 샘플링 (전체는 너무 많음)

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )
])

print("="*70)
print("Left vs Right Context Vector Extraction")
print("="*70)
print()

# ============================================================================
# Step 1: Load Model
# ============================================================================

print(f"[1/5] Loading checkpoint...")
print(f"  Path: {CHECKPOINT_PATH}")

if not Path(CHECKPOINT_PATH).exists():
    print(f"  ❌ Checkpoint not found!")
    print(f"  Available checkpoints:")
    for ckpt in Path("runs/mobile_vla_no_chunk_20251209").rglob("*.ckpt"):
        print(f"    - {ckpt}")
    exit(1)

# Load model (simplified - will use inference engine)
# For now, we'll extract from saved context if available
# Or we need to implement proper model loading

print(f"  ✅ Checkpoint found")
print()

# ============================================================================
# Step 2: Load Episodes
# ============================================================================

print(f"[2/5] Loading episodes...")

left_files = sorted(DATASET_PATH.glob("*left*.h5"))
right_files = sorted(DATASET_PATH.glob("*right*.h5"))

print(f"  Left episodes: {len(left_files)}")
print(f"  Right episodes: {len(right_files)}")

# Sample
left_files = left_files[:MAX_EPISODES_PER_DIR]
right_files = right_files[:MAX_EPISODES_PER_DIR]

print(f"  Sampling: {MAX_EPISODES_PER_DIR} each")
print()

# ============================================================================
# Step 3: Extract Context Vectors - Left
# ============================================================================

print(f"[3/5] Extracting Left context vectors...")

left_contexts = []

for h5_file in tqdm(left_files, desc="Left"):
    try:
        with h5py.File(h5_file, 'r') as f:
            images = f['images']
            
            # Get first 8 frames
            frames = []
            for i in range(min(WINDOW_SIZE, len(images))):
                img = Image.fromarray(images[i].astype(np.uint8))
                frames.append(transform(img))
            
            # Pad if needed
            while len(frames) < WINDOW_SIZE:
                frames.append(frames[-1])
            
            # Stack: [8, 3, 224, 224]
            frames_tensor = torch.stack(frames)
            
            # TODO: Extract context from model
            # For now, use placeholder
            # context = model.encode_images(frames_tensor)
            # context shape: [8, 64, 2048]
            
            # Placeholder (will be replaced by actual extraction)
            context = np.random.randn(WINDOW_SIZE, 64, 2048)
            
            left_contexts.append(context)
            
    except Exception as e:
        print(f"  ⚠️  Error processing {h5_file.name}: {e}")
        continue

left_contexts = np.array(left_contexts)
print(f"  Shape: {left_contexts.shape}")
print()

# ============================================================================
# Step 4: Extract Context Vectors - Right
# ============================================================================

print(f"[4/5] Extracting Right context vectors...")

right_contexts = []

for h5_file in tqdm(right_files, desc="Right"):
    try:
        with h5py.File(h5_file, 'r') as f:
            images = f['images']
            
            frames = []
            for i in range(min(WINDOW_SIZE, len(images))):
                img = Image.fromarray(images[i].astype(np.uint8))
                frames.append(transform(img))
            
            while len(frames) < WINDOW_SIZE:
                frames.append(frames[-1])
            
            frames_tensor = torch.stack(frames)
            
            # TODO: Extract context from model
            context = np.random.randn(WINDOW_SIZE, 64, 2048)
            
            right_contexts.append(context)
            
    except Exception as e:
        print(f"  ⚠️  Error processing {h5_file.name}: {e}")
        continue

right_contexts = np.array(right_contexts)
print(f"  Shape: {right_contexts.shape}")
print()

# ============================================================================
# Step 5: Save
# ============================================================================

print(f"[5/5] Saving context vectors...")

left_path = OUTPUT_DIR / "left_contexts.npy"
right_path = OUTPUT_DIR / "right_contexts.npy"

np.save(left_path, left_contexts)
np.save(right_path, right_contexts)

print(f"  ✅ Saved: {left_path}")
print(f"  ✅ Saved: {right_path}")

# Save metadata
metadata = {
    'left': {
        'shape': list(left_contexts.shape),
        'n_episodes': len(left_files),
        'files': [str(f.name) for f in left_files[:10]]  # First 10
    },
    'right': {
        'shape': list(right_contexts.shape),
        'n_episodes': len(right_files),
        'files': [str(f.name) for f in right_files[:10]]
    },
    'parameters': {
        'window_size': WINDOW_SIZE,
        'checkpoint': str(CHECKPOINT_PATH),
        'max_episodes': MAX_EPISODES_PER_DIR
    }
}

with open(OUTPUT_DIR / "metadata.json", 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"  ✅ Saved: metadata.json")
print()

print("="*70)
print("✅ Context vector extraction complete!")
print("="*70)
print()
print("Next steps:")
print("  1. Run: python3 docs/latent_space_analysis/compare_left_right.py")
print("  2. Run: python3 docs/latent_space_analysis/visualize_latent_space.py")
print()
print("⚠️  NOTE: This script uses PLACEHOLDER context vectors.")
print("   Real extraction requires loading the trained model.")
print("   See: docs/WEDNESDAY_MEETING_PLAN.md for full implementation.")
