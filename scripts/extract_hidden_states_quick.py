#!/usr/bin/env python3
"""
긴급: Hidden States 추출 (5분 안에!)
Case 5 Best Model에서 Left vs Right latent space 추출
"""

import torch
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json

print("="*60)
print("Hidden States 추출 시작!")
print("="*60)
print()

# ============================================================================
# Config
# ============================================================================

CHECKPOINT = "runs/mobile_vla_no_chunk_20251209/lightning_logs/version_0/checkpoints/epoch_epoch=04-val_loss=val_loss=0.001.ckpt"
DATA_DIR = "ROS_action/mobile_vla_dataset"
OUTPUT_DIR = Path("docs/meeting_20251210/latent_space_results")
OUTPUT_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# 빠른 샘플링 (시간 부족)
NUM_SAMPLES = 10  # Left 10, Right 10

# ============================================================================
# Load Model
# ============================================================================

print()
print("1. Loading checkpoint...")

try:
    checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)
    print(f"  ✅ Checkpoint loaded")
    print(f"  Keys: {list(checkpoint.keys())[:5]}")
except Exception as e:
    print(f"  ❌ Error: {e}")
    print()
    print("대안: 직접 Kosmos-2 로드")
    
    from transformers import AutoProcessor, AutoModelForVision2Seq
    
    model_name = ".vlms/kosmos-2-patch14-224"
    print(f"  Loading {model_name}...")
    
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForVision2Seq.from_pretrained(model_name)
    model = model.to(DEVICE)
    model.eval()
    
    print(f"  ✅ Kosmos-2 loaded")

# ============================================================================
# Load Data
# ============================================================================

print()
print("2. Loading data samples...")

import glob
from PIL import Image

left_files = sorted(glob.glob(f"{DATA_DIR}/*left*.h5"))[:NUM_SAMPLES]
right_files = sorted(glob.glob(f"{DATA_DIR}/*right*.h5"))[:NUM_SAMPLES]

print(f"  Left: {len(left_files)} files")
print(f"  Right: {len(right_files)} files")

# ============================================================================
# Extract Hidden States
# ============================================================================

print()
print("3. Extracting hidden states...")

def extract_hidden_states(h5_file, model, processor):
    """H5 파일에서 hidden states 추출"""
    
    with h5py.File(h5_file, 'r') as f:
        # 첫 이미지
        img_array = f['images'][0]
        img = Image.fromarray(img_array.astype(np.uint8))
        
        # Instruction
        instr_bytes = f['language_instruction'][0]
        instruction = instr_bytes.decode('utf-8') if isinstance(instr_bytes, bytes) else str(instr_bytes)
    
    # Process
    inputs = processor(text=instruction, images=img, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    # Forward (no grad)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    # Hidden states (last layer)
    hidden_states = outputs.hidden_states[-1]  # (1, seq_len, hidden_dim)
    
    # Mean pooling
    vector = hidden_states.mean(dim=1).cpu().numpy()  # (1, hidden_dim)
    
    return vector[0]

# Left
left_vectors = []
print("  Left episodes:")
for f in tqdm(left_files, desc="  "):
    try:
        vec = extract_hidden_states(f, model, processor)
        left_vectors.append(vec)
    except Exception as e:
        print(f"    Skip {f}: {e}")

# Right
right_vectors = []
print("  Right episodes:")
for f in tqdm(right_files, desc="  "):
    try:
        vec = extract_hidden_states(f, model, processor)
        right_vectors.append(vec)
    except Exception as e:
        print(f"    Skip {f}: {e}")

left_vectors = np.array(left_vectors)
right_vectors = np.array(right_vectors)

print()
print(f"  ✅ Left: {left_vectors.shape}")
print(f"  ✅ Right: {right_vectors.shape}")

# ============================================================================
# Cosine Similarity
# ============================================================================

print()
print("4. Computing cosine similarity...")

from sklearn.metrics.pairwise import cosine_similarity

# Intra-class
sim_LL = cosine_similarity(left_vectors).mean()
sim_RR = cosine_similarity(right_vectors).mean()

# Inter-class
sim_LR = cosine_similarity(left_vectors, right_vectors).mean()

# Separation
separation = (sim_LL + sim_RR) / 2 - sim_LR

print(f"  Left-Left:   {sim_LL:.4f}")
print(f"  Right-Right: {sim_RR:.4f}")
print(f"  Left-Right:  {sim_LR:.4f}")
print(f"  Separation:  {separation:.4f}")

# ============================================================================
# Save Results
# ============================================================================

print()
print("5. Saving results...")

# Vectors
np.save(OUTPUT_DIR / "left_vectors.npy", left_vectors)
np.save(OUTPUT_DIR / "right_vectors.npy", right_vectors)

# Metrics
results = {
    "similarity": {
        "left_left": float(sim_LL),
        "right_right": float(sim_RR),
        "left_right": float(sim_LR),
        "separation": float(separation)
    },
    "num_samples": {
        "left": len(left_vectors),
        "right": len(right_vectors)
    },
    "model": CHECKPOINT
}

with open(OUTPUT_DIR / "results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"  ✅ Saved to {OUTPUT_DIR}/")
print()
print("="*60)
print("완료!")
print("="*60)
print()
print(f"결과: Separation = {separation:.4f}")
if separation > 0.2:
    print("✅ Left vs Right가 구분됨!")
else:
    print("⚠️  구분이 약함")
