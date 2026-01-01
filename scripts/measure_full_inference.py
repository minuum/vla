#!/usr/bin/env python3
"""
전체 파이프라인 추론 메모리 측정 스크립트
Jetson 로컬 온디바이스 실행을 위한 RAM 리소스 분석 (Peak 메모리 포함)
"""

import torch
import psutil
import time
import json
from pathlib import Path
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import numpy as np

def get_memory_info():
    """통합 메모리 정보 (CPU+GPU)"""
    mem = psutil.virtual_memory()
    gpu_allocated = torch.cuda.memory_allocated(0) / 1024**3 if torch.cuda.is_available() else 0
    gpu_reserved = torch.cuda.memory_reserved(0) / 1024**3 if torch.cuda.is_available() else 0
    
    return {
        "ram_total_gb": round(mem.total / 1024**3, 2),
        "ram_used_gb": round(mem.used / 1024**3, 2),
        "ram_available_gb": round(mem.available / 1024**3, 2),
        "ram_percent": round(mem.percent, 1),
        "gpu_allocated_gb": round(gpu_allocated, 2),
        "gpu_reserved_gb": round(gpu_reserved, 2)
    }

def main():
    print("="*70)
    print("  전체 파이프라인 추론 메모리 측정 (FP16)")
    print("="*70)
    print()
    
    # 베이스라인
    print("📊 [1/6] 베이스라인 측정...")
    baseline = get_memory_info()
    print(f"   RAM: {baseline['ram_used_gb']}/{baseline['ram_total_gb']} GB")
    print(f"   GPU: {baseline['gpu_allocated_gb']} GB")
    print()
    
    # 모델 경로
    model_path = ".vlms/kosmos-2-patch14-224"
    
    # 전체 모델 로딩
    print("📦 [2/6] 전체 모델 로딩 중...")
    start_time = time.time()
    
    try:
        model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map={"": 0},
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        
        load_time = time.time() - start_time
        print(f"   ✅ 로딩 완료 ({load_time:.1f}초)")
        print()
        
    except Exception as e:
        print(f"   ❌ 실패: {e}")
        return
    
    # 로딩 후 메모리
    print("📊 [3/6] 로딩 후 메모리...")
    after_load = get_memory_info()
    print(f"   RAM: {after_load['ram_used_gb']}/{after_load['ram_total_gb']} GB")
    print(f"   GPU: {after_load['gpu_allocated_gb']} GB")
    print()
    
    # 더미 이미지 생성
    print("🖼️  [4/6] 테스트 이미지 생성...")
    dummy_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    text_prompt = "Navigate to the target"
    print("   ✅ 이미지 준비 완료")
    print()
    
    # 추론 실행
    print("🚀 [5/6] 추론 실행 중...")
    try:
        inputs = processor(images=dummy_image, text=text_prompt, return_tensors="pt")
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10)
        
        print("   ✅ 추론 완료")
        print()
        
    except Exception as e:
        print(f"   ❌ 추론 실패: {e}")
        import traceback
        traceback.print_exc()
    
    # Peak 메모리
    print("📊 [6/6] Peak 메모리...")
    peak_mem = get_memory_info()
    print(f"   RAM: {peak_mem['ram_used_gb']}/{peak_mem['ram_total_gb']} GB")
    print(f"   GPU: {peak_mem['gpu_allocated_gb']} GB")
    print()
    
    # 증가량 요약
    print("="*70)
    print("📈 메모리 증가량 요약")
    print("="*70)
    
    load_ram = after_load['ram_used_gb'] - baseline['ram_used_gb']
    load_gpu = after_load['gpu_allocated_gb'] - baseline['gpu_allocated_gb']
    peak_ram = peak_mem['ram_used_gb'] - baseline['ram_used_gb']
    peak_gpu = peak_mem['gpu_allocated_gb'] - baseline['gpu_allocated_gb']
    
    print(f"모델 로딩:")
    print(f"  RAM: +{load_ram:.2f} GB")
    print(f"  GPU: +{load_gpu:.2f} GB")
    print()
    print(f"추론 Peak:")
    print(f"  RAM: +{peak_ram:.2f} GB")
    print(f"  GPU: +{peak_gpu:.2f} GB")
    print()
    print(f"Activation 증가: +{peak_ram - load_ram:.2f} GB")
    print()
    
    # 파라미터 정보
    total_params = sum(p.numel() for p in model.parameters())
    print(f"ℹ️  모델 정보")
    print(f"   Parameters: {total_params:,} ({total_params/1e9:.2f}B)")
    print()
    
    # 결과 저장
    result = {
        "model": "Full Pipeline (Kosmos-2)",
        "dtype": "FP16",
        "baseline": baseline,
        "after_load": after_load,
        "peak": peak_mem,
        "increase": {
            "load_ram_gb": round(load_ram, 2),
            "load_gpu_gb": round(load_gpu, 2),
            "peak_ram_gb": round(peak_ram, 2),
            "peak_gpu_gb": round(peak_gpu, 2),
            "activation_gb": round(peak_ram - load_ram, 2)
        },
        "parameters": total_params,
        "load_time_sec": round(load_time, 1)
    }
    
    # 저장
    output_dir = Path("logs")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "memory_full_inference_fp16.json"
    
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"💾 결과 저장: {output_file}")
    print()
    print("="*70)
    print("✅ 전체 파이프라인 메모리 측정 완료!")
    print("="*70)

if __name__ == "__main__":
    main()
