#!/usr/bin/env python3
"""
PaliGemma-3B Ablation Test for English Instructions
Tests if the model can distinguish LEFT vs RIGHT instructions
"""

import os
import sys
import torch
import numpy as np
from PIL import Image

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Mobile_VLA'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'RoboVLMs_upstream'))

from Mobile_VLA.inference_pipeline import MobileVLAInferencePipeline

# PaliGemma checkpoint path (update after training)
CHECKPOINT_PATH = "runs/mobile_vla_paligemma/paligemma/mobile_vla_paligemma_finetune/2026-01-07/mobile_vla_paligemma_lora/epoch_epoch=00-val_loss=val_loss=0.040.ckpt"

# English instructions
INSTRUCTIONS = {
    "LEFT": "Navigate around the obstacle on the left side and reach the cup",
    "RIGHT": "Navigate around the obstacle on the right side and reach the cup",
}

# Test image (dummy)
def create_test_image():
    """Create a dummy test image"""
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(img)

def main():
    print("=" * 70)
    print("ABLATION TEST: PaliGemma-3B English Instruction")
    print("=" * 70)
    print()
    
    print("Loading model...")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    
    # Load model
    pipeline = MobileVLAInferencePipeline(
        model_path=CHECKPOINT_PATH,
        config_path="Mobile_VLA/configs/mobile_vla_paligemma_lora.json"
    )
    
    print("Model loaded successfully")
    print()
    
    # Test image
    test_image = create_test_image()
    
    results = {}
    
    # Test both instructions
    for label, instruction in INSTRUCTIONS.items():
        print(f"[{label}] {instruction}")
        
        # Predict
        action = pipeline.predict(test_image, instruction)
        
        # Extract linear_y (첫 번째 window, 첫 번째 chunk)
        if isinstance(action, tuple):
            action = action[0]
        
        print(f"  Action shape: {action.shape}")
        
        # Handle (window_size, chunk_size, 2) format
        if len(action.shape) == 3:
            linear_y = float(action[0, 0, 1])  # First window, first chunk, y-axis
        elif len(action.shape) == 2:
            linear_y = float(action[0, 1])  # First step, y-axis
        else:
            linear_y = float(action[1])  # y-axis
        
        print(f"  linear_y: {linear_y:.4f}")
        print()
        
        results[label] = linear_y
    
    # Results
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"LEFT  → {results['LEFT']:.4f}")
    print(f"RIGHT → {results['RIGHT']:.4f}")
    print(f"Diff: {abs(results['LEFT'] - results['RIGHT']):.4f}")
    print()
    
    # Evaluation
    print("=" * 70)
    if results['LEFT'] > 0 and results['RIGHT'] < 0:
        print("✓ SUCCESS: Model correctly interprets instructions!")
        print("  LEFT  → positive (left turn)")
        print("  RIGHT → negative (right turn)")
    elif abs(results['LEFT'] - results['RIGHT']) < 0.1:
        print("✗ FAIL: IGNORES instructions")
        print("  Both outputs are nearly identical")
    else:
        print("⚠ PARTIAL: Outputs differ but not as expected")
        print(f"  LEFT: {results['LEFT']:.4f}, RIGHT: {results['RIGHT']:.4f}")

if __name__ == "__main__":
    main()
