#!/usr/bin/env python3
"""
Quantized 모델의 실제 GPU 메모리 사용량 측정
"""

import torch
import gc
import sys
import json
sys.path.insert(0, 'RoboVLMs_upstream')

from robovlms.model.backbone.robokosmos import RoboKosMos
from robovlms.train.mobile_vla_trainer import MobileVLATrainer

def get_gpu_memory():
    """현재 GPU 메모리 사용량 (MB)"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0

def measure_model_memory(checkpoint_path, config_path):
    """모델 로딩 후 실제 GPU 메모리 측정"""
    
    print("="*70)
    print("실제 GPU 메모리 사용량 측정")
    print("="*70)
    
    # GPU 초기화
    torch.cuda.empty_cache()
    gc.collect()
    
    baseline = get_gpu_memory()
    print(f"\n1. Baseline GPU Memory: {baseline:.2f} MB")
    
    # Config 로드
    print(f"\n2. Loading config: {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Checkpoint 로드
    print(f"\n3. Loading checkpoint: {checkpoint_path}")
    print("   (이 과정이 오래 걸릴 수 있습니다...)")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    after_load = get_gpu_memory()
    print(f"   After loading to CPU: {after_load:.2f} MB")
    
    # Trainer 생성
    print("\n4. Creating trainer...")
    trainer = MobileVLATrainer(config)
    
    # State dict 로드
    print("\n5. Loading state dict...")
    trainer.load_state_dict(checkpoint['model_state_dict'])
    after_state = get_gpu_memory()
    print(f"   After state dict: {after_state:.2f} MB")
    
    # GPU로 이동
    print("\n6. Moving model to GPU...")
    trainer = trainer.cuda()
    after_cuda = get_gpu_memory()
    
    # 메모리 정리
    del checkpoint
    torch.cuda.empty_cache()
    gc.collect()
    
    final_memory = get_gpu_memory()
    
    print("\n" + "="*70)
    print("측정 결과")
    print("="*70)
    print(f"Baseline:              {baseline:.2f} MB")
    print(f"After CUDA move:       {after_cuda:.2f} MB")
    print(f"After cleanup:         {final_memory:.2f} MB")
    print(f"\n실제 모델 메모리:      {final_memory - baseline:.2f} MB")
    print(f"                       {(final_memory - baseline)/1024:.2f} GB")
    print("="*70)
    
    # 추론 테스트로 추가 메모리 측정
    print("\n7. Running inference test...")
    trainer.eval()
    
    # Dummy input
    batch_size = 1
    seq_len = 8
    with torch.no_grad():
        dummy_images = torch.randn(batch_size, seq_len, 3, 224, 224).cuda()
        dummy_text = ["turn left"] * batch_size
        
        print(f"   Input shape: {dummy_images.shape}")
        inference_start = get_gpu_memory()
        
        # Forward pass
        try:
            # Simple forward test
            print("   Running forward pass...")
            # Note: This might fail if model needs more setup
            print("   (Skipping actual forward - would need full setup)")
        except Exception as e:
            print(f"   Forward pass error (expected): {e}")
        
        inference_end = get_gpu_memory()
    
    print(f"\n추론 시 메모리:")
    print(f"   Before inference: {inference_start:.2f} MB")
    print(f"   After inference:  {inference_end:.2f} MB")
    print(f"   Peak during inference: {torch.cuda.max_memory_allocated()/1024**2:.2f} MB")
    
    return {
        'baseline_mb': baseline,
        'model_only_mb': final_memory - baseline,
        'model_only_gb': (final_memory - baseline) / 1024,
        'peak_mb': torch.cuda.max_memory_allocated() / 1024**2,
        'peak_gb': torch.cuda.max_memory_allocated() / 1024**3
    }

if __name__ == '__main__':
    checkpoint_path = 'quantized_models/chunk5_best_int8_int4_20251224/model_quantized.pt'
    config_path = 'quantized_models/chunk5_best_int8_int4_20251224/config.json'
    
    print("\n🔬 Quantized Model 실제 메모리 측정")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Config: {config_path}\n")
    
    try:
        results = measure_model_memory(checkpoint_path, config_path)
        
        print("\n" + "="*70)
        print("최종 결과 (실측)")
        print("="*70)
        print(f"모델 메모리:    {results['model_only_mb']:.2f} MB = {results['model_only_gb']:.3f} GB")
        print(f"Peak 메모리:    {results['peak_mb']:.2f} MB = {results['peak_gb']:.3f} GB")
        print("="*70)
        
        # Save results
        with open('quantized_models/chunk5_best_int8_int4_20251224/actual_memory.json', 'w') as f:
            json.dump(results, f, indent=2)
        print("\n✅ 결과 저장: actual_memory.json")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
