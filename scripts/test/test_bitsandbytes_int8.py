#!/usr/bin/env python3
"""
BitsAndBytes INT8 Quantization 적용
OpenVLA, BitVLA와 같은 방법
"""

import torch
import sys
import time

print("="*70)
print("BitsAndBytes INT8 Quantization (VLA 표준)")
print("="*70)

# Step 1: Check BitsAndBytes installation
print("\n1. Checking BitsAndBytes installation...")
try:
    import bitsandbytes as bnb
    print(f"   ✅ BitsAndBytes version: {bnb.__version__}")
except ImportError:
    print("   ❌ BitsAndBytes not installed")
    print("   Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "bitsandbytes", "accelerate"])
    import bitsandbytes as bnb
    print(f"   ✅ Installed: {bnb.__version__}")

# Step 2: Load model with BitsAndBytes
print("\n2. Loading model with BitsAndBytes INT8...")

sys.path.insert(0, 'RoboVLMs_upstream')

from transformers import BitsAndBytesConfig
from robovlms.train.mobile_vla_trainer import MobileVLATrainer
import json

# Config
config_path = 'Mobile_VLA/configs/mobile_vla_chunk5_20251217.json'
checkpoint_path = 'runs/mobile_vla_no_chunk_20251209/kosmos/mobile_vla_finetune/2025-12-17/mobile_vla_chunk5_20251217/epoch_epoch=06-val_loss=val_loss=0.067.ckpt'

with open(config_path, 'r') as f:
    config = json.load(f)

# BitsAndBytes Config (OpenVLA style)
print("\n3. Creating BitsAndBytes config...")
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=False,
    llm_int8_threshold=6.0
)

print(f"   Config: {bnb_config}")

# Load model
print("\n4. Loading model with quantization...")

# PROBLEM: MobileVLATrainer doesn't directly support BitsAndBytesConfig
# We need to modify the underlying model loading

# Alternative: Load base model with BitsAndBytes, then wrap with trainer
from transformers import AutoModelForVision2Seq, AutoProcessor

print("   Loading Kosmos-2 with BitsAndBytes...")

try:
    # Load processor
    processor = AutoProcessor.from_pretrained(
        config['vlm']['pretrained_model_name_or_path']
    )
    
    # Load model with BitsAndBytes
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated() / 1024**3
    
    model = AutoModelForVision2Seq.from_pretrained(
        config['vlm']['pretrained_model_name_or_path'],
        quantization_config=bnb_config,
        device_map="auto",  # Auto GPU allocation
        torch_dtype=torch.float16
    )
    
    gpu_mem = torch.cuda.memory_allocated() / 1024**3
    model_size = gpu_mem - baseline
    
    print(f"   ✅ Model loaded with BitsAndBytes")
    print(f"   GPU Memory: {model_size:.3f} GB")
    
    # Step 5: Test inference
    print("\n5. Testing inference...")
    
    # Dummy input
    from PIL import Image
    import numpy as np
    
    dummy_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    dummy_text = "<grounding>Move forward"
    
    inputs = processor(
        text=dummy_text,
        images=dummy_image,
        return_tensors="pt"
    ).to(model.device)
    
    # Inference
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20
        )
    latency = (time.time() - start) * 1000
    
    print(f"   ✅ Inference successful")
    print(f"   Latency: {latency:.1f} ms")
    
    # Decode output
    decoded = processor.batch_decode(outputs, skip_special_tokens=True)
    print(f"   Output: {decoded[0][:100]}")
    
    print("\n" + "="*70)
    print("✅ BitsAndBytes INT8 Success!")
    print("="*70)
    print(f"GPU Memory: {model_size:.3f} GB")
    print(f"Latency: {latency:.1f} ms")
    print("\n비교:")
    print(f"  PyTorch Static INT8: 6.3 GB, 15s")
    print(f"  BitsAndBytes INT8: {model_size:.1f} GB, {latency/1000:.1f}s")
    print(f"  절감: {(1 - model_size/6.3)*100:.1f}%")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    
    print("\n⚠️  Note:")
    print("MobileVLATrainer may need modification to support BitsAndBytes")
    print("Next step: Modify RoboKosMos backbone to accept quantization_config")
