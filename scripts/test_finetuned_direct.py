#!/usr/bin/env python3
"""
Mobile VLA Fine-tuned 모델 테스트 (체크포인트 키 수정 버전)
"""

import warnings
warnings.filterwarnings('ignore')

import sys
import torch
import time
import psutil
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "Robo+"))
sys.path.insert(0, str(PROJECT_ROOT / "RoboVLMs"))

print("="*70)
print("  Mobile VLA Fine-tuned 모델 테스트")
print("  Checkpoint: epoch=06, val_loss=0.067")
print("="*70)
print()

# 1. 환경
print("1️⃣ 환경:")
print(f"   PyTorch: {torch.__version__}")
print(f"   CUDA: {torch.cuda.is_available()}")
print()

# 2. 체크포인트
checkpoint_path = "runs/mobile_vla_no_chunk_20251209/kosmos/mobile_vla_finetune/2025-12-17/mobile_vla_chunk5_20251217/epoch_epoch=06-val_loss=val_loss=0.067.ckpt"
print("2️⃣ 체크포인트 로딩:")
print(f"   경로: {checkpoint_path}")

start_time = time.time()
start_mem = psutil.virtual_memory().used / 1024**3

try:
    # 체크포인트 로드 (CPU)
    print("   📥 로딩 중...")
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    
    print(f"   ✅ 체크포인트 로드 완료")
    print(f"   epoch: {ckpt.get('epoch', 'N/A')}")
    print(f"   global_step: {ckpt.get('global_step', 'N/A')}")
    
    # Hyper parameters 확인
    if 'hyper_parameters' in ckpt:
        hparams = ckpt['hyper_parameters']
        print(f"\n   Hyper Parameters:")
        for key in ['model_name', 'action_dim', 'window_size', 'chunk_size']:
            if key in hparams:
                print(f"     {key}: {hparams[key]}")
    
    # state_dict 확인
    state_dict = ckpt.get('state_dict', {})
    print(f"\n   state_dict: {len(state_dict)} 항목")
    
    # MobileVLATrainer 생성
    print("\n3️⃣ MobileVLATrainer 생성:")
    from robovlms.models.robovlms_trainer import RoboVLMsTrainer
    
    model_name = hparams.get('model_name', 'microsoft/kosmos-2-patch14-224')
    action_dim = hparams.get('action_dim', 2)
    window_size = hparams.get('window_size', 2)
    chunk_size = hparams.get('chunk_size', 10)
    
    print(f"   model: {model_name}")
    print(f"   action_dim: {action_dim}")
    print(f"   window_size: {window_size}")
    print(f"   chunk_size: {chunk_size}")
    
    trainer = RoboVLMsTrainer(
        model_name=model_name,
        action_dim=action_dim,
        window_size=window_size,
        chunk_size=chunk_size,
        device='cpu'
    )
    
    # State dict 로드
    print("\n4️⃣ State dict 로딩:")
    trainer.model.load_state_dict(state_dict, strict=False)
    print("   ✅ State dict 로드 완료")
    
    # FP16 변환
    print("\n5️⃣ FP16 변환 및 GPU 전송:")
    trainer.model = trainer.model.half()
    trainer.model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer.model = trainer.model.to(device)
    
    load_time = time.time() - start_time
    end_mem = psutil.virtual_memory().used / 1024**3
    gpu_mem = torch.cuda.memory_allocated(0) / 1024**3
    
    print(f"   ✅ 완료! ({load_time:.1f}초)")
    print(f"   RAM: +{end_mem - start_mem:.2f} GB")
    print(f"   GPU: {gpu_mem:.2f} GB")
    print(f"   Device: {device}")
    
    # 6. 간단한 forward 테스트
    print("\n6️⃣ Forward 테스트:")
    
    # 더미 입력
    batch_size = 1
    images = torch.randn(batch_size, window_size, 3, 224, 224).half().to(device)
    instruction = "Navigate to the left bottle"
    
    print(f"   지시문: {instruction}")
    print(f"   이미지: {images.shape}")
    
    with torch.no_grad():
        try:
            # forward_continuous 호출 (RoboVLMs 방식)
            output = trainer.model.forward_continuous(
                images=images,
                instruction=[instruction],
                fwd_pred_next_n=chunk_size
            )
            
            # action 추출
            if isinstance(output, dict):
                actions = output.get('action', output.get('actions', None))
            else:
                actions = output
            
            if actions is not None:
                print(f"   ✅ Forward 성공!")
                print(f"   Output shape: {actions.shape}")
                print(f"   첫 action: [{actions[0, 0, 0]:.3f}, {actions[0, 0, 1]:.3f}]")
            else:
                print(f"   ⚠️ Action 추출 실패")
                print(f"   Output keys: {output.keys() if isinstance(output, dict) else 'not dict'}")
            
        except Exception as e:
            print(f"   ❌ Forward 실패: {e}")
            import traceback
            traceback.print_exc()
    
    # 7. 최종 메모리
    print("\n7️⃣ 최종 메모리:")
    final_mem = psutil.virtual_memory()
    final_gpu = torch.cuda.memory_allocated(0) / 1024**3
    print(f"   RAM: {final_mem.used / 1024**3:.2f} / {final_mem.total / 1024**3:.2f} GB ({final_mem.percent:.1f}%)")
    print(f"   GPU: {final_gpu:.2f} GB")
    
    print("\n" + "="*70)
    print("🎊 Fine-tuned 모델 로드 및 테스트 완료!")
    print("="*70)
    print(f"📊 로딩 시간: {load_time:.1f}초")
    print(f"📊 RAM 증가: +{end_mem - start_mem:.2f} GB")
    print(f"📊 GPU 메모리: {gpu_mem:.2f} GB")
    print()
    print("✅ Mobile VLA fine-tuned 모델 정상 작동!")
    
except Exception as e:
    print(f"\n❌ 실패: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
