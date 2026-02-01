#!/usr/bin/env python3
"""
Chunk10 Best Model RMSE 계산
"""

import torch
import json
from pathlib import Path

def calculate_chunk10_rmse():
    """Chunk10 best checkpoint의 RMSE 계산"""
    
    # Chunk10 best checkpoint (실제 경로)
    best_ckpt = Path("/home/billy/25-1kp/vla/runs/mobile_vla_no_chunk_20251209/kosmos/mobile_vla_finetune/2025-12-17/mobile_vla_chunk10_20251217/epoch_epoch=05-val_loss=val_loss=0.284.ckpt")
    
    if not best_ckpt.exists():
        print(f"⚠️  Checkpoint를 찾을 수 없습니다: {best_ckpt}")
        return None
    
    print(f"✅ Checkpoint 발견: {best_ckpt.name}")
    
    # Checkpoint 로드 (메타데이터만)
    try:
        checkpoint = torch.load(best_ckpt, map_location='cpu')
        
        # Validation loss 추출
        val_loss = None
        if 'val_loss' in checkpoint:
            val_loss = checkpoint['val_loss']
        elif 'callbacks' in checkpoint:
            # Lightning callbacks에서 추출
            val_loss = checkpoint.get('callbacks', {}).get('val_loss')
        
        if val_loss is not None:
            # RMSE = sqrt(val_loss) (MSE 기준)
            import math
            rmse = math.sqrt(val_loss)
            
            result = {
                "checkpoint": best_ckpt.name,
                "val_loss": float(val_loss),
                "rmse": float(rmse),
                "epoch": 5
            }
            
            print(f"\n📊 Chunk10 Metrics:")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            
            # 결과 저장
            output_file = Path("/home/billy/25-1kp/vla/docs/chunk10_metrics.json")
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            print(f"\n✅ 결과 저장: {output_file}")
            return result
        else:
            print("⚠️  Checkpoint에서 val_loss를 찾을 수 없습니다.")
            print("\nCheckpoint keys:")
            for key in checkpoint.keys():
                print(f"  - {key}")
            return None
            
    except Exception as e:
        print(f"❌ Checkpoint 로드 실패: {e}")
        return None

if __name__ == "__main__":
    print("="*60)
    print("Chunk10 RMSE 계산")
    print("="*60)
    
    result = calculate_chunk10_rmse()
    
    if result:
        print("\n" + "="*60)
        print("✅ 계산 완료!")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("⚠️  수동으로 val_loss를 확인해야 합니다.")
        print("="*60)
