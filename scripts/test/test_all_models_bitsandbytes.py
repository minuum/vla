#!/usr/bin/env python3
"""
BitsAndBytes INT8 추론 테스트 (All Models)
Chunk5, Left Chunk10, Right Chunk10 모델별 테스트
"""

import torch
import json
import sys
import time
import os
from transformers import BitsAndBytesConfig

sys.path.insert(0, 'RoboVLMs_upstream')

print("="*70)
print("BitsAndBytes INT8 - All Models Test")
print("="*70)

# BitsAndBytes Config
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0
)

# Test models
models = [
    {
        'name': 'Chunk5 Best',
        'config': 'Mobile_VLA/configs/mobile_vla_chunk5_20251217.json',
        'checkpoint': 'runs/mobile_vla_no_chunk_20251209/kosmos/mobile_vla_finetune/2025-12-17/mobile_vla_chunk5_20251217/epoch_epoch=06-val_loss=val_loss=0.067.ckpt',
        'val_loss': 0.067
    },
    {
        'name': 'Left Chunk10 Best',
        'config': 'Mobile_VLA/configs/mobile_vla_left_chunk10_20251217.json',
        'checkpoint': 'runs/mobile_vla_no_chunk_20251209/kosmos/mobile_vla_finetune/2025-12-17/mobile_vla_left_chunk10_20251217/epoch_epoch=07-val_loss=val_loss=0.010.ckpt',
        'val_loss': 0.010
    },
    {
        'name': 'Right Chunk10 Best',
        'config': 'Mobile_VLA/configs/mobile_vla_right_chunk5_20251217.json',
        'checkpoint': 'runs/mobile_vla_no_chunk_20251209/kosmos/mobile_vla_finetune/2025-12-17/mobile_vla_right_chunk5_20251217/epoch_epoch=07-val_loss=val_loss=0.013.ckpt',
        'val_loss': 0.013
    }
]

results = []

for model_info in models:
    print(f"\n{'='*70}")
    print(f"Testing: {model_info['name']}")
    print(f"{'='*70}")
    
    try:
        # Check files exist
        if not os.path.exists(model_info['config']):
            print(f"   ❌ Config not found: {model_info['config']}")
            continue
            
        if not os.path.exists(model_info['checkpoint']):
            print(f"   ❌ Checkpoint not found: {model_info['checkpoint']}")
            continue
        
        # Load config
        with open(model_info['config'], 'r') as f:
            config = json.load(f)
        
        # Load checkpoint
        print(f"\n1. Loading checkpoint...")
        checkpoint = torch.load(model_info['checkpoint'], map_location='cpu')
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            print(f"   ❌ No state_dict found")
            continue
        
        print(f"   ✅ Loaded {len(state_dict)} parameters")
        
        # Create model with BitsAndBytes
        print(f"\n2. Creating model with BitsAndBytes INT8...")
        from robovlms.train.mobile_vla_trainer import MobileVLATrainer
        
        torch.cuda.empty_cache()
        baseline = torch.cuda.memory_allocated() / 1024**3
        
        model = MobileVLATrainer(
            config,
            quantization_config=bnb_config
        )
        
        # Load weights
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        model = model.cuda()
        
        gpu_mem = torch.cuda.memory_allocated() / 1024**3
        model_size = gpu_mem - baseline
        
        print(f"   ✅ GPU Memory: {model_size:.3f} GB")
        
        # Test inference
        print(f"\n3. Testing inference...")
        
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
        
        # Store results
        results.append({
            'name': model_info['name'],
            'val_loss': model_info['val_loss'],
            'gpu_memory': model_size,
            'latency': latency,
            'status': 'SUCCESS'
        })
        
        print(f"\n✅ {model_info['name']}: Success")
        print(f"   Memory: {model_size:.2f} GB, Latency: {latency:.0f} ms")
        
        # Clean up
        del model
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"\n❌ {model_info['name']}: Failed")
        print(f"   Error: {e}")
        
        results.append({
            'name': model_info['name'],
            'val_loss': model_info['val_loss'],
            'gpu_memory': 0,
            'latency': 0,
            'status': 'FAILED',
            'error': str(e)
        })

# Final Report
print(f"\n{'='*70}")
print("FINAL RESULTS")
print(f"{'='*70}\n")

print(f"{'Model':<25} {'Val Loss':<12} {'GPU Mem':<12} {'Latency':<12} {'Status':<10}")
print(f"{'-'*25} {'-'*12} {'-'*12} {'-'*12} {'-'*10}")

for r in results:
    if r['status'] == 'SUCCESS':
        print(f"{r['name']:<25} {r['val_loss']:<12.3f} {r['gpu_memory']:<12.2f} {r['latency']:<12.0f} {r['status']:<10}")
    else:
        print(f"{r['name']:<25} {r['val_loss']:<12.3f} {'N/A':<12} {'N/A':<12} {r['status']:<10}")

success_count = sum(1 for r in results if r['status'] == 'SUCCESS')
print(f"\n✅ Success: {success_count}/{len(results)} models")

if success_count > 0:
    avg_memory = sum(r['gpu_memory'] for r in results if r['status'] == 'SUCCESS') / success_count
    avg_latency = sum(r['latency'] for r in results if r['status'] == 'SUCCESS') / success_count
    print(f"📊 Average: {avg_memory:.2f} GB, {avg_latency:.0f} ms")
