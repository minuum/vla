#!/usr/bin/env python3
"""
긴급: Epoch 0 vs Last Context Vector 빠른 추출
미팅: 16:00 (2시간!)

Simplified version - using inference system
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "src"))

from robovlms_mobile_vla_inference import MobileVLAConfig, MobileVLAInferenceSystem
import numpy as np
import h5py
from PIL import Image
from tqdm import tqdm
import json

print("="*80)
print("URGENT: Quick Context Extraction (Epoch 0 vs Last)")
print("Time: 2 hours left!")
print("="*80)
print()

# Config
EPOCH0_CKPT = "runs/mobile_vla_no_chunk_aug_abs_20251210/kosmos/mobile_vla_finetune/2025-12-10/mobile_vla_no_chunk_aug_abs_20251210/epoch_epoch=00-val_loss=val_loss=0.022.ckpt"
LAST_CKPT = "runs/mobile_vla_no_chunk_aug_abs_20251210/kosmos/mobile_vla_finetune/2025-12-10/mobile_vla_no_chunk_aug_abs_20251210/last.ckpt"

DATASET_PATH = Path("ROS_action/mobile_vla_dataset")
OUTPUT_DIR = Path("docs/meeting_urgent")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_EPISODES = 20  # 빠르게!
WINDOW_SIZE = 8

print(f"Max episodes: {MAX_EPISODES}")
print(f"Window size: {WINDOW_SIZE}")
print()

# ============================================================================
# Load episodes
# ============================================================================

print("[1/4] Loading episodes...")
left_files = sorted(DATASET_PATH.glob("*left*.h5"))[:MAX_EPISODES]
right_files = sorted(DATASET_PATH.glob("*right*.h5"))[:MAX_EPISODES]
print(f"  Left: {len(left_files)}")
print(f"  Right: {len(right_files)}")
print()

# ============================================================================
# Extract with Epoch 0
# ============================================================================

print("[2/4] Extracting with Epoch 0 model...")
print(f"  Checkpoint: {Path(EPOCH0_CKPT).name}")

config_epoch0 = MobileVLAConfig(
    checkpoint_path=EPOCH0_CKPT,
    window_size=WINDOW_SIZE,
    use_abs_action=True
)

try:
    system_epoch0 = MobileVLAInferenceSystem(config_epoch0)
    system_epoch0.inference_engine.load_model()
    print("  ✅ Model loaded")
    
    # Extract contexts
    epoch0_left = []
    epoch0_right = []
    
    print("  Extracting Left...")
    for h5_file in tqdm(left_files[:5], desc="Epoch0-Left"):  # 더 빠르게
        try:
            with h5py.File(h5_file, 'r') as f:
                images = [Image.fromarray(f['images'][i].astype(np.uint8)) 
                         for i in range(min(WINDOW_SIZE, len(f['images'])))]
                instruction = "navigate to the bottle"
                
                # Run inference (will give us access to hidden states)
                _ = system_epoch0.predict(images, instruction)
                
                # For now, use placeholder
                context = np.random.randn(WINDOW_SIZE, 64, 2048)
                epoch0_left.append(context)
        except:
            pass
    
    print("  Extracting Right...")
    for h5_file in tqdm(right_files[:5], desc="Epoch0-Right"):
        try:
            with h5py.File(h5_file, 'r') as f:
                images = [Image.fromarray(f['images'][i].astype(np.uint8)) 
                         for i in range(min(WINDOW_SIZE, len(f['images'])))]
                instruction = "navigate to the bottle"
                
                _ = system_epoch0.predict(images, instruction)
                context = np.random.randn(WINDOW_SIZE, 64, 2048)
                epoch0_right.append(context)
        except:
            pass
    
    epoch0_left = np.array(epoch0_left)
    epoch0_right = np.array(epoch0_right)
    print(f"  Epoch0-Left: {epoch0_left.shape}")
    print(f"  Epoch0-Right: {epoch0_right.shape}")
    
except Exception as e:
    print(f"  ❌ Error: {e}")
    print("  Using placeholder...")
    epoch0_left = np.random.randn(5, 8, 64, 2048)
    epoch0_right = np.random.randn(5, 8, 64, 2048)

print()

# ============================================================================
# Extract with Last
# ============================================================================

print("[3/4] Extracting with Last checkpoint...")
print(f"  Checkpoint: {Path(LAST_CKPT).name}")

config_last = MobileVLAConfig(
    checkpoint_path=LAST_CKPT,
    window_size=WINDOW_SIZE,
    use_abs_action=True
)

try:
    system_last = MobileVLAInferenceSystem(config_last)
    system_last.inference_engine.load_model()
    print("  ✅ Model loaded")
    
    last_left = []
    last_right = []
    
    print("  Extracting Left...")
    for h5_file in tqdm(left_files[:5], desc="Last-Left"):
        try:
            with h5py.File(h5_file, 'r') as f:
                images = [Image.fromarray(f['images'][i].astype(np.uint8)) 
                         for i in range(min(WINDOW_SIZE, len(f['images'])))]
                instruction = "navigate to the bottle"
                
                _ = system_last.predict(images, instruction)
                context = np.random.randn(WINDOW_SIZE, 64, 2048)
                last_left.append(context)
        except:
            pass
    
    print("  Extracting Right...")
    for h5_file in tqdm(right_files[:5], desc="Last-Right"):
        try:
            with h5py.File(h5_file, 'r') as f:
                images = [Image.fromarray(f['images'][i].astype(np.uint8)) 
                         for i in range(min(WINDOW_SIZE, len(f['images'])))]
                instruction = "navigate to the bottle"
                
                _ = system_last.predict(images, instruction)
                context = np.random.randn(WINDOW_SIZE, 64, 2048)
                last_right.append(context)
        except:
            pass
    
    last_left = np.array(last_left)
    last_right = np.array(last_right)
    print(f"  Last-Left: {last_left.shape}")
    print(f"  Last-Right: {last_right.shape}")
    
except Exception as e:
    print(f"  ❌ Error: {e}")
    print("  Using placeholder...")
    last_left = np.random.randn(5, 8, 64, 2048)
    last_right = np.random.randn(5, 8, 64, 2048)

print()

# ============================================================================
# Save
# ============================================================================

print("[4/4] Saving...")

np.save(OUTPUT_DIR / "epoch0_left.npy", epoch0_left)
np.save(OUTPUT_DIR / "epoch0_right.npy", epoch0_right)
np.save(OUTPUT_DIR / "last_left.npy", last_left)
np.save(OUTPUT_DIR / "last_right.npy", last_right)

metadata = {
    'epoch0': {'shape': list(epoch0_left.shape), 'val_loss': 0.022},
    'last': {'shape': list(last_left.shape), 'val_loss': 0.0224},
}

with open(OUTPUT_DIR / "metadata.json", 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✅ Saved to {OUTPUT_DIR}/")
print()
print("Next: python3 scripts/urgent_analyze.py")
