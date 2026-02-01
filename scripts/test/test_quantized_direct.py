#!/usr/bin/env python3
"""
PyTorch JIT로 진짜 INT8 Inference 실행
TorchScript quantized model - 메모리도 INT8
"""

import torch
import sys
import time

sys.path.insert(0, 'RoboVLMs_upstream')

print("="*70)
print("PyTorch JIT INT8 Inference (진짜 INT8)")
print("="*70)

# Load INT8 model
int8_path = 'quantized_models/chunk5_best_int8/model.pt'
print(f"\n1. Loading INT8 model...")

int8_checkpoint = torch.load(int8_path, map_location='cpu')
config = int8_checkpoint['config']
state_dict = int8_checkpoint['model_state_dict']

print(f"   Quantized params: {sum(1 for v in state_dict.values() if hasattr(v, 'is_quantized') and v.is_quantized)}")

# Create model architecture with quantization support
print("\n2. Creating quantized model...")

from robovlms.train.mobile_vla_trainer import MobileVLATrainer
from torch.quantization import get_default_qconfig, prepare, convert

# Create base model
model = MobileVLATrainer(config)

# Set qconfigs (same as before)
def set_qconfig_recursive(module):
    for name, child in module.named_children():
        if isinstance(child, torch.nn.Embedding):
            from torch.quantization import float_qparams_weight_only_qconfig
            child.qconfig = float_qparams_weight_only_qconfig
        elif isinstance(child, (torch.nn.Linear, torch.nn.Conv2d)):
            child.qconfig = get_default_qconfig('fbgemm')
        else:
            set_qconfig_recursive(child)

set_qconfig_recursive(model)
model.qconfig = get_default_qconfig('fbgemm')

# Prepare and convert to quantized model
print("   Preparing quantized model...")
model = prepare(model, inplace=True)

# Calibrate with dummy data (필요)
print("   Calibrating...")
with torch.no_grad():
    dummy_vision = torch.randn(1, 8, 3, 224, 224)
    dummy_lang = torch.ones(1, 256, dtype=torch.long)
    dummy_attention = torch.ones(1, 256, dtype=torch.bool)
    
    for i in range(3):
        try:
            _ = model.model.forward(dummy_vision, dummy_lang, attention_mask=dummy_attention)
        except:
            pass

print("   Converting to INT8...")
model_quantized = convert(model, inplace=True)

# Load INT8 weights
print("   Loading INT8 weights...")
model_quantized.load_state_dict(state_dict, strict=False)
model_quantized.eval()

# Measure memory BEFORE moving to GPU
print("\n3. Memory measurement...")

# CPU memory
import psutil
import os
process = psutil.Process(os.getpid())
cpu_mem = process.memory_info().rss / 1024**3
print(f"   CPU Memory: {cpu_mem:.2f} GB")

# GPU memory
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated() / 1024**3
    
    model_quantized = model_quantized.cuda()
    
    gpu_mem = torch.cuda.memory_allocated() / 1024**3
    model_size = gpu_mem - baseline
    
    print(f"   GPU Memory (quantized): {model_size:.3f} GB")
    
    # Test inference on GPU
    print("\n4. Testing GPU inference...")
    
    with torch.no_grad():
        dummy_vision = torch.randn(1, 8, 3, 224, 224).cuda()
        dummy_lang = torch.ones(1, 256, dtype=torch.long).cuda()
        dummy_attention = torch.ones(1, 256, dtype=torch.bool).cuda()
        
        start = time.time()
        try:
            output = model_quantized.model.inference(
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
        
        except Exception as e:
            print(f"   ❌ Error: {e}")
else:
    print("   GPU not available")

print("\n" + "="*70)
print("완료")
print("="*70)
print(f"GPU Memory: {model_size:.3f} GB")
print("\n⚠️  현재 PyTorch quantization은 storage만 INT8")
print("실제 inference는 여전히 FP32로 실행됩니다.")
print("\n진짜 INT8 inference를 위해서는:")
print("1. TensorRT 사용 (NVIDIA GPU)")
print("2. ONNX Runtime (QDQ format)")
print("3. 또는 Jetson에서 직접 테스트")
