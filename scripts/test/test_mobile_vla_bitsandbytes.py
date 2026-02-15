#!/usr/bin/env python3
"""
Mobile VLA with BitsAndBytes INT8 Quantization
OpenVLA/BitVLA 표준 방법 적용
"""

import torch
import json
import sys
import time
import os

sys.path.insert(0, 'RoboVLMs_upstream')

print("="*70)
print("Mobile VLA + BitsAndBytes INT8 (VLA 표준)")
print("="*70)

# BitsAndBytes Config
print("\n1. Creating BitsAndBytes config...")
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,  # INT8 quantization
    llm_int8_enable_fp32_cpu_offload=False,
    llm_int8_threshold=6.0
)

print(f"   ✅ Config: INT8, threshold=6.0")

# Load config
config_path = 'Mobile_VLA/configs/mobile_vla_chunk5_20251217.json'
checkpoint_path = 'runs/mobile_vla_no_chunk_20251209/kosmos/mobile_vla_finetune/2025-12-17/mobile_vla_chunk5_20251217/epoch_epoch=06-val_loss=val_loss=0.067.ckpt'

with open(config_path, 'r') as f:
    config = json.load(f)

print(f"\n2. Loading Mobile VLA with BitsAndBytes...")
print(f"   Config: {config_path}")
print(f"   Checkpoint: {os.path.basename(checkpoint_path)}")

# Load model with quantization
from robovlms.train.mobile_vla_trainer import MobileVLATrainer

try:
    # Create model with BitsAndBytes
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated() / 1024**3
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Create model with quantization config
    model = MobileVLATrainer(
        config,
        quantization_config=bnb_config  # Pass BitsAndBytes config
    )
    
    # Load weights
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        raise ValueError("No state_dict found")
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    # Move to GPU and measure
    model = model.cuda()
    
    gpu_mem = torch.cuda.memory_allocated() / 1024**3
    model_size = gpu_mem - baseline
    
    print(f"\n3. Memory Usage:")
    print(f"   GPU Memory: {model_size:.3f} GB")
    
    # Test inference
    print(f"\n4. Testing inference...")
    
    with torch.no_grad():
        dummy_vision = torch.randn(1, 8, 3, 224, 224).cuda()
        dummy_lang = torch.ones(1, 256, dtype=torch.long).cuda()
        dummy_attention = torch.ones(1, 256, dtype=torch.bool).cuda()
        
        start = time.time()
        output = model.model.inference(
            vision_x=dummy_vision,
            lang_x=dummy_lang,
            attention_mask=dummy_attention
        )
        latency = (time.time() - start) * 1000
        
        print(f"   ✅ Inference successful")
        print(f"   Latency: {latency:.1f} ms")
        
        if isinstance(output, dict) and 'action' in output:
            action = output['action']
            if isinstance(action, tuple):
                action = action[0]
            print(f"   Action shape: {action.shape}")
    
    print("\n" + "="*70)
    print("✅ Mobile VLA + BitsAndBytes Success!")
    print("="*70)
    print(f"GPU Memory: {model_size:.3f} GB   (vs FP32: 6.3GB)")
    print(f"Latency: {latency/1000:.2f}s      (vs FP32: 15s)")
    print(f"Reduction: {(1 - model_size/6.3)*100:.1f}%")
    print(f"Speedup: {15000/latency:.1f}x")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    
    print("\n⚠️  Troubleshooting:")
    print("1. BitsAndBytes installed? pip install bitsandbytes")
    print("2. CUDA available? torch.cuda.is_available()")
    print("3. GPU memory sufficient? (need ~2GB free)")
