#!/usr/bin/env python3
"""
Pretrained 모델 메모리 측정 (FP16 - BitsAndBytes 우회)
"""

import sys
import time
import torch
from pathlib import Path
import psutil

print("="*70)
print("  Pretrained 모델 메모리 측정 (FP16)")
print("  (BitsAndBytes 문제로 INT8 대신 FP16 사용)")
print("="*70)
print()

def check_memory():
    """메모리 체크"""
    mem = psutil.virtual_memory()
    gpu_mem = torch.cuda.memory_allocated(0) / 1024**3 if torch.cuda.is_available() else 0
    return {
        "ram_gb": round(mem.used / 1024**3, 2),
        "gpu_gb": round(gpu_mem, 2)
    }

# 1. 시작 메모리
print("📊 시작 메모리:")
start_mem = check_memory()
print(f"   RAM: {start_mem['ram_gb']} GB")
print(f"   GPU: {start_mem['gpu_gb']} GB")
print()

# 2. 모델 경로 확인
model_path = ".vlms/kosmos-2-patch14-224"
if not Path(model_path).exists():
    print(f"❌ 모델 경로 없음: {model_path}")
    sys.exit(1)

# 3. 모델 로딩 (FP16)
print(f"📦 모델 로딩 시작: {model_path}")
print("   모드: FP16 (torch_dtype=float16)")
print("   (메모리 절약을 위해 INT8 대신 FP16 사용)")
print()

try:
    from transformers import AutoModelForVision2Seq
    
    start_time = time.time()
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=torch.float16,  # FP16 사용
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    load_time = time.time() - start_time
    
    print(f"   ✅ 모델 로드 성공! ({load_time:.1f}초)")
    print()
    
    # 4. 로딩 후 메모리
    print("📊 로딩 후 메모리:")
    after_mem = check_memory()
    print(f"   RAM: {after_mem['ram_gb']} GB (+{after_mem['ram_gb'] - start_mem['ram_gb']:.2f} GB)")
    print(f"   GPU: {after_mem['gpu_gb']} GB (+{after_mem['gpu_gb'] - start_mem['gpu_gb']:.2f} GB)")
    print()
    
    # 5. 모델 정보
    print("ℹ️  모델 정보:")
    print(f"   Device: {model.device}")
    print(f"   dtype: {next(model.parameters()).dtype}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {total_params:,}")
    
    # 메모리 예상치
    param_memory_gb = total_params * 2 / 1024**3  # FP16 = 2 bytes
    print(f"   예상 메모리 (FP16): {param_memory_gb:.2f} GB")
    print()
    
    print("="*70)
    print("✅ FP16 모델 로딩 성공!")
    print()
    print("📝 참고:")
    print("   - BitsAndBytes INT8: CUDA 커널 호환성 문제로 사용 불가")
    print("   - FP16: Jetson에서 안정적으로 동작")
    print("   - 메모리: INT8(1GB) < FP16(2GB) < FP32(4GB)")
    print("="*70)
    
except Exception as e:
    print(f"❌ 모델 로드 실패: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
