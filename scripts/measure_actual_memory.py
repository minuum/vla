#!/usr/bin/env python3
"""
간단한 GPU 메모리 실측 - 원본 vs Quantized 비교
"""

import torch
import gc
import os

def measure_model_size(checkpoint_path):
    """파일에서 직접 로딩하여 GPU 메모리 측정"""
    
    # GPU 초기화
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    
    baseline = torch.cuda.memory_allocated() / 1024**3
    print(f"Baseline: {baseline:.3f} GB")
    
    # Checkpoint 로드
    print(f"Loading: {checkpoint_path}")
    print("(loading to GPU directly...)")
    
    checkpoint = torch.load(checkpoint_path, map_location='cuda:0')
    
    after_load = torch.cuda.memory_allocated() / 1024**3
    peak = torch.cuda.max_memory_allocated() / 1024**3
    
    print(f"After load: {after_load:.3f} GB")
    print(f"Peak: {peak:.3f} GB")
    print(f"Model size (actual): {after_load - baseline:.3f} GB")
    
    del checkpoint
    torch.cuda.empty_cache()
    
    return after_load - baseline, peak

print("="*70)
print("실제 GPU 메모리 측정 (A5000)")
print("="*70)

# 원본 모델
print("\n1. 원본 모델 (FP32)")
print("-"*70)
orig_size, orig_peak = measure_model_size(
    'runs/mobile_vla_no_chunk_20251209/kosmos/mobile_vla_finetune/2025-12-17/mobile_vla_chunk5_20251217/epoch_epoch=06-val_loss=val_loss=0.067.ckpt'
)

print("\n\n2. Quantized 모델 (INT8 + INT4)")
print("-"*70)
quant_size, quant_peak = measure_model_size(
    'quantized_models/chunk5_best_int8_int4_20251224/model_quantized.pt'
)

print("\n" + "="*70)
print("비교 결과 (실측)")
print("="*70)
print(f"원본 FP32:     {orig_size:.3f} GB")
print(f"Quantized:     {quant_size:.3f} GB")
print(f"절감:          {orig_size - quant_size:.3f} GB ({(1 - quant_size/orig_size)*100:.1f}%)")
print("="*70)

# Save결과
import json
results = {
    'original_gb': float(orig_size),
    'quantized_gb': float(quant_size),
    'reduction_gb': float(orig_size - quant_size),
    'reduction_percent': float((1 - quant_size/orig_size) * 100)
}

with open('quantized_models/chunk5_best_int8_int4_20251224/actual_memory.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n✅ 결과 저장: actual_memory.json")
