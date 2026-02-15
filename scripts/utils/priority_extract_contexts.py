#!/usr/bin/env python3
"""
실제 Context Vector 추출 (Inference 기반)
==========================================

Priority 1 작업: Fine-Tuned vs No Fine-Tuning

Step 1: Fine-Tuned (Case 5, Epoch 4) context 추출
Step 2: Pre-trained (초기 상태) context 추출  
Step 3: 비교 분석
"""

import torch
import numpy as np
import h5py
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import sys
import json

# Add path
sys.path.append(str(Path.cwd() / "RoboVLMs_upstream"))

print("="*80)
print("Priority Task: Extract FT vs NoFT Context Vectors")
print("="*80)
print()

# ============================================================================
# Configuration
# ============================================================================

FINETUNED_CHECKPOINT = Path("runs/mobile_vla_no_chunk_20251209/kosmos/mobile_vla_finetune/2025-12-09/mobile_vla_no_chunk_20251209/epoch_epoch=04-val_loss=val_loss=0.001.ckpt")
DATASET_PATH = Path("ROS_action/mobile_vla_dataset")
OUTPUT_DIR = Path("docs/latent_space_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_EPISODES = 50
WINDOW_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {DEVICE}")
print(f"Fine-tuned checkpoint: {FINETUNED_CHECKPOINT.name}")
print(f"Max episodes per direction: {MAX_EPISODES}")
print()

# ============================================================================
# Helper: Load episodes
# ============================================================================

def load_episode_files(pattern, max_count):
    """Load H5 files matching pattern"""
    files = sorted(DATASET_PATH.glob(pattern))
    print(f"  Found: {len(files)} files matching '{pattern}'")
    selected = files[:min(max_count, len(files))]
    print(f"  Selected: {len(selected)} files")
    return selected

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )
])

def extract_frames(h5_file):
    """Extract and transform frames from H5"""
    with h5py.File(h5_file, 'r') as f:
        images = f['images']
        frames = []
        
        for i in range(min(WINDOW_SIZE, len(images))):
            img = Image.fromarray(images[i].astype(np.uint8))
            frames.append(transform(img))
        
        # Pad if needed
        while len(frames) < WINDOW_SIZE:
            frames.append(frames[-1])
        
        return torch.stack(frames)  # [8, 3, 224, 224]

# ============================================================================
# Step 1: Load episodes
# ============================================================================

print("[1/3] Loading episode files...")
left_files = load_episode_files("*left*.h5", MAX_EPISODES)
right_files = load_episode_files("*right*.h5", MAX_EPISODES)
print()

# ============================================================================
# Step 2: Extract Fine-Tuned contexts (Case 5)
# ============================================================================

print("[2/3] Extracting Fine-Tuned contexts (Case 5)...")
print(f"  Checkpoint: {FINETUNED_CHECKPOINT.name}")
print(f"  Size: {FINETUNED_CHECKPOINT.stat().st_size / 1e9:.1f} GB")
print()

# TODO: Load actual checkpoint and extract
# For now, using test_inference_stepbystep.py approach

print("  Option 1: Use test_inference_stepbystep.py")
print("    → Already has model loading code")
print("    → Can extract hidden states")
print()

print("  Option 2: Direct PyTorch Lightning loading")
print("    → Load checkpoint")
print("    → Extract vision encoder output")
print()

print("  ⚠️  Using placeholder for now - implement based on option 1 or 2")

# Placeholder
finetuned_left = np.random.randn(len(left_files), WINDOW_SIZE, 64, 2048)
finetuned_right = np.random.randn(len(right_files), WINDOW_SIZE, 64, 2048)

print(f"  FT-Left shape: {finetuned_left.shape}")
print(f"  FT-Right shape: {finetuned_right.shape}")
print()

# ============================================================================
# Step 3: Extract Pre-trained contexts
# ============================================================================

print("[3/3] Extracting Pre-trained contexts...")
print("  Loading Kosmos-2 pre-trained (no LoRA weights)")
print()

print("  Option 1: transformers.AutoModel")
print("    from transformers import AutoModel")
print("    model = AutoModel.from_pretrained('microsoft/kosmos-2')")
print()

print("  Option 2: RoboVLMs initial state")
print("    → Load config")
print("    → Initialize model (random LoRA)")
print()

print("  ⚠️  Using placeholder for now")

# Placeholder
pretrained_left = np.random.randn(len(left_files), WINDOW_SIZE, 64, 2048)
pretrained_right = np.random.randn(len(right_files), WINDOW_SIZE, 64, 2048)

print(f"  NoFT-Left shape: {pretrained_left.shape}")
print(f"  NoFT-Right shape: {pretrained_right.shape}")
print()

# ============================================================================
# Save
# ============================================================================

print("Saving context vectors...")

outputs = {
    'FT5_left.npy': finetuned_left,
    'FT5_right.npy': finetuned_right,
    'noFT_left.npy': pretrained_left,
    'noFT_right.npy': pretrained_right,
}

for filename, data in outputs.items():
    path = OUTPUT_DIR / filename
    np.save(path, data)
    print(f"  ✅ {filename}: {data.shape}")

# Metadata
metadata = {
    'fine_tuned': {
        'checkpoint': str(FINETUNED_CHECKPOINT),
        'model': 'Case 5 Epoch 4',
        'left_shape': list(finetuned_left.shape),
        'right_shape': list(finetuned_right.shape),
    },
    'pre_trained': {
        'source': 'Kosmos-2 (no fine-tuning)',
        'left_shape': list(pretrained_left.shape),
        'right_shape': list(pretrained_right.shape),
    },
    'parameters': {
        'window_size': WINDOW_SIZE,
        'max_episodes': MAX_EPISODES,
        'device': DEVICE,
    }
}

with open(OUTPUT_DIR / "extraction_metadata.json", 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"  ✅ extraction_metadata.json")
print()

print("="*80)
print("✅ Extraction complete (placeholder)")
print("="*80)
print()
print("⚠️  NEXT STEPS:")
print("1. Implement actual model loading (use test_inference_stepbystep.py)")
print("2. Extract real context vectors")
print("3. Run: python3 scripts/compare_ft_noFT.py")
print("4. Run: python3 scripts/visualize_comparison.py")
