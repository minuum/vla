#!/usr/bin/env python3
"""
모든 체크포인트의 MAE 값 확인 스크립트
"""

import torch
import os
from pathlib import Path

def check_checkpoint(checkpoint_path):
    """체크포인트 정보 확인"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        info = {
            'path': checkpoint_path,
            'mae': checkpoint.get('val_mae', 'N/A'),
            'epoch': checkpoint.get('epoch', 'N/A'),
            'model_keys': len(checkpoint.get('model_state_dict', {})),
            'file_size_mb': os.path.getsize(checkpoint_path) / (1024 * 1024)
        }
        
        # 모델 타입 판별
        state_dict = checkpoint.get('model_state_dict', {})
        kosmos_keys = [key for key in state_dict.keys() if 'kosmos' in key.lower()]
        clip_keys = [key for key in state_dict.keys() if 'clip' in key.lower()]
        
        if len(clip_keys) > 0 and len(kosmos_keys) > 0:
            info['model_type'] = 'Kosmos2+CLIP Hybrid'
        elif len(kosmos_keys) > 0:
            info['model_type'] = 'Pure Kosmos2'
        elif len(clip_keys) > 0:
            info['model_type'] = 'CLIP Only'
        else:
            info['model_type'] = 'Unknown'
        
        return info
        
    except Exception as e:
        print(f"❌ 체크포인트 로드 실패 {checkpoint_path}: {e}")
        return None

def main():
    # 체크포인트 경로들
    checkpoint_paths = [
        "results/simple_lstm_results_extended/final_simple_lstm_model.pth",
        "results/simple_clip_lstm_results_extended/best_simple_clip_lstm_model.pth",
        "results/mobile_vla_epoch_1.pt",
        "results/mobile_vla_epoch_2.pt",
        "results/mobile_vla_epoch_3.pt",
        "models/experimental/simplified_robovlms_best.pth"
    ]
    
    print("🔍 모든 체크포인트 MAE 확인")
    print("="*80)
    
    results = []
    
    for checkpoint_path in checkpoint_paths:
        if not os.path.exists(checkpoint_path):
            print(f"⚠️ 파일 없음: {checkpoint_path}")
            continue
            
        info = check_checkpoint(checkpoint_path)
        if info:
            results.append(info)
            print(f"✅ {Path(checkpoint_path).name}")
            print(f"   모델 타입: {info['model_type']}")
            print(f"   MAE: {info['mae']}")
            print(f"   에포크: {info['epoch']}")
            print(f"   모델 키: {info['model_keys']}개")
            print(f"   파일 크기: {info['file_size_mb']:.2f} MB")
            print()
    
    # MAE 기준 정렬
    mae_results = [r for r in results if r['mae'] != 'N/A']
    mae_results.sort(key=lambda x: x['mae'])
    
    print("🎯 MAE 기준 순위")
    print("="*40)
    for i, result in enumerate(mae_results, 1):
        print(f"{i}. {Path(result['path']).name} ({result['model_type']})")
        print(f"   MAE: {result['mae']}")
        print(f"   에포크: {result['epoch']}")
        print()

if __name__ == "__main__":
    main()
