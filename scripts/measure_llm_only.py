#!/usr/bin/env python3
"""
LLM 메모리 측정 스크립트
Jetson 로컬 온디바이스 실행을 위한 RAM 리소스 분석
"""

import torch
import psutil
import time
import json
from pathlib import Path
from transformers import AutoModelForCausalLM

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
    print("  LLM 메모리 측정 (FP16)")
    print("="*70)
    print()
    
    # 베이스라인
    print("📊 [1/4] 베이스라인 측정...")
    baseline = get_memory_info()
    print(f"   RAM: {baseline['ram_used_gb']}/{baseline['ram_total_gb']} GB")
    print(f"   GPU: {baseline['gpu_allocated_gb']} GB")
    print()
    
    # 모델 경로
    model_path = ".vlms/kosmos-2-patch14-224"
    
    # LLM 로딩
    print("📦 [2/4] LLM 로딩 중...")
    print("   (Kosmos-2 text_model)")
    start_time = time.time()
    
    try:
        # Kosmos-2의 Text Model (LLM)만 로드
        from transformers import AutoModel
        full_model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        llm = full_model.text_model.to('cuda')
        
        load_time = time.time() - start_time
        print(f"   ✅ 로딩 완료 ({load_time:.1f}초)")
        print()
        
    except Exception as e:
        print(f"   ❌ 실패: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 로딩 후 메모리
    print("📊 [3/4] 로딩 후 메모리...")
    after_load = get_memory_info()
    print(f"   RAM: {after_load['ram_used_gb']}/{after_load['ram_total_gb']} GB")
    print(f"   GPU: {after_load['gpu_allocated_gb']} GB")
    print()
    
    # 증가량
    print("📈 [4/4] 메모리 증가량")
    ram_inc = after_load['ram_used_gb'] - baseline['ram_used_gb']
    gpu_inc = after_load['gpu_allocated_gb'] - baseline['gpu_allocated_gb']
    print(f"   RAM: +{ram_inc:.2f} GB")
    print(f"   GPU: +{gpu_inc:.2f} GB")
    print(f"   통합 메모리 실제 사용: ~{ram_inc:.2f} GB")
    print()
    
    # 파라미터 정보
    total_params = sum(p.numel() for p in llm.parameters())
    print(f"ℹ️  모델 정보")
    print(f"   Parameters: {total_params:,} ({total_params/1e9:.2f}B)")
    print(f"   dtype: {next(llm.parameters()).dtype}")
    print()
    
    # 결과 저장
    result = {
        "model": "LLM (Kosmos-2 Text Model)",
        "dtype": "FP16",
        "baseline": baseline,
        "after_load": after_load,
        "increase": {
            "ram_gb": round(ram_inc, 2),
            "gpu_gb": round(gpu_inc, 2)
        },
        "parameters": total_params,
        "load_time_sec": round(load_time, 1)
    }
    
    # 저장
    output_dir = Path("logs")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "memory_llm_fp16.json"
    
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"💾 결과 저장: {output_file}")
    print()
    print("="*70)
    print("✅ LLM 메모리 측정 완료!")
    print("="*70)

if __name__ == "__main__":
    main()
