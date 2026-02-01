#!/usr/bin/env python3
import sys
sys.path.insert(0, "/home/billy/25-1kp/vla")
sys.path.insert(0, "/home/billy/25-1kp/vla/RoboVLMs_upstream")

from Mobile_VLA.inference_pipeline import MobileVLAInferencePipeline
from PIL import Image
import numpy as np

checkpoint = "/home/billy/25-1kp/vla/runs/mobile_vla_no_chunk_20251209/kosmos/mobile_vla_finetune/2026-01-07/mobile_vla_chunk5_20251217/epoch_epoch=01-val_loss=val_loss=0.354.ckpt"
config = "Mobile_VLA/configs/mobile_vla_chunk5_20251217.json"

print("=" * 70)
print("ABLATION TEST: English Instruction (Epoch 1)")
print("=" * 70)

print("\nLoading model...")
pipeline = MobileVLAInferencePipeline(checkpoint, config, device="cuda")

test_image = Image.new('RGB', (224, 224), color='green')

instructions = {
    'LEFT': "Navigate around the obstacle on the left side and reach the cup",
    'RIGHT': "Navigate around the obstacle on the right side and reach the cup"
}

results = {}
for label, instruction in instructions.items():
    print(f"\n[{label}] {instruction}")
    result = pipeline.predict(test_image, instruction)
    action = np.array(result['action'])
    
    print(f"  Action shape: {action.shape}")
    print(f"  Action (first): {action[0, 0] if action.ndim == 3 else (action[0] if action.ndim == 2 else action)}")
    
    # Extract first action's linear_y
    if action.ndim == 3:
        # Format: (window_size, chunk_size, action_dim) → use first window, first chunk
        linear_y = float(action[0, 0, 1])
    elif action.ndim == 2:
        # Format: (chunk_size, action_dim) → use first chunk
        linear_y = float(action[0, 1])
    else:
        # Format: (action_dim,) → use second element
        linear_y = float(action[1])
    
    results[label] = linear_y
    print(f"  linear_y: {results[label]:.4f}")

print(f"\n{'='*70}")
print("RESULTS")
print("="*70)
print(f"LEFT  → {results['LEFT']:.4f}")
print(f"RIGHT → {results['RIGHT']:.4f}")
print(f"Diff: {abs(results['LEFT'] - results['RIGHT']):.4f}")

print(f"\n{'='*70}")
if results['LEFT'] > 0 and results['RIGHT'] < 0:
    print("✓ SUCCESS: Instructions work!")
elif results['LEFT'] * results['RIGHT'] < 0:
    print("⚠ PARTIAL: Some sensitivity")
else:
    print("✗ FAIL: IGNORES instructions")
