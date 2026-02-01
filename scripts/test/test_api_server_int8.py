#!/usr/bin/env python3
"""
Test API Server with BitsAndBytes INT8
"""

import sys
sys.path.insert(0, 'Mobile_VLA')

from inference_server import MobileVLAInference
import torch

print("="*70)
print("API Server BitsAndBytes INT8 Test")
print("="*70)

# Model config
checkpoint_path = 'runs/mobile_vla_no_chunk_20251209/kosmos/mobile_vla_finetune/2025-12-17/mobile_vla_chunk5_20251217/epoch_epoch=06-val_loss=val_loss=0.067.ckpt'
config_path = 'Mobile_VLA/configs/mobile_vla_chunk5_20251217.json'

try:
    print("\n1. Initializing API Server with INT8...")
    server = MobileVLAInference(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        device='cuda'
    )
    
    print("\n2. Testing inference...")
    from PIL import Image
    import numpy as np
    import base64
    import io
    
    # Create dummy image
    dummy_img = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
    
    # Convert to base64
    buffered = io.BytesIO()
    dummy_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    # Test predict
    import time
    start = time.time()
    result = server.predict(
        image_base64=img_str,
        instruction="Move forward",
        strategy="receding_horizon"
    )
    latency = (time.time() - start) * 1000
    
    print(f"\n3. Result:")
    print(f"   Action: {result['action']}")
    print(f"   Latency: {result['latency_ms']:.1f} ms")
    print(f"   Strategy: {result['strategy']}")
    print(f"   Source: {result['source']}")
    
    print("\n" + "="*70)
    print("✅ API Server INT8 Test SUCCESS!")
    print("="*70)
    print(f"Total Latency: {latency:.1f} ms")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
