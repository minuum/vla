#!/usr/bin/env python3
"""
모든 모델 체크포인트 분석 및 성능 비교
"""

import os
import torch
import json
from pathlib import Path
import numpy as np

def analyze_checkpoint(checkpoint_path):
    """체크포인트 분석"""
    try:
        if not os.path.exists(checkpoint_path):
            return None
            
        # 체크포인트 로드
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 기본 정보 추출
        info = {
            'path': checkpoint_path,
            'file_size_mb': os.path.getsize(checkpoint_path) / (1024 * 1024),
            'model_keys': len(checkpoint.keys()) if isinstance(checkpoint, dict) else 0
        }
        
        # 모델 타입 추정
        if 'model_state_dict' in checkpoint:
            model_keys = list(checkpoint['model_state_dict'].keys())
            if 'clip_model' in str(model_keys):
                info['model_type'] = 'CLIP-based'
            elif 'kosmos2' in str(model_keys):
                info['model_type'] = 'Kosmos2-based'
            elif 'lstm' in str(model_keys):
                info['model_type'] = 'LSTM-based'
            else:
                info['model_type'] = 'Unknown'
        else:
            info['model_type'] = 'Unknown'
        
        # 성능 메트릭 추출
        if 'val_mae' in checkpoint:
            info['mae'] = checkpoint['val_mae']
        elif 'mae' in checkpoint:
            info['mae'] = checkpoint['mae']
        else:
            info['mae'] = 'N/A'
            
        if 'epoch' in checkpoint:
            info['epoch'] = checkpoint['epoch']
        else:
            info['epoch'] = 'N/A'
            
        if 'val_loss' in checkpoint:
            info['val_loss'] = checkpoint['val_loss']
        elif 'loss' in checkpoint:
            info['loss'] = checkpoint['loss']
        else:
            info['val_loss'] = 'N/A'
            
        if 'train_loss' in checkpoint:
            info['train_loss'] = checkpoint['train_loss']
        else:
            info['train_loss'] = 'N/A'
            
        # 모델 정보
        if 'model_info' in checkpoint:
            info['model_info'] = checkpoint['model_info']
        else:
            info['model_info'] = {}
            
        return info
        
    except Exception as e:
        print(f"❌ 체크포인트 분석 실패 {checkpoint_path}: {e}")
        return None

def main():
    # 모든 체크포인트 경로
    checkpoint_paths = [
        # 최신 Enhanced 모델들
        "./checkpoints_enhanced/best_model_epoch_3.pt",
        "./checkpoints_enhanced/best_model_epoch_2.pt", 
        "./checkpoints_enhanced/best_model_epoch_1.pt",
        
        # 기존 모델들
        "./Robo+/Mobile_VLA/results/simple_lstm_results_extended/final_simple_lstm_model.pth",
        "./Robo+/Mobile_VLA/results/simple_clip_lstm_results_extended/best_simple_clip_lstm_model.pth",
        "./Robo+/Mobile_VLA/original_clip_augmented_results/best_original_clip_augmented_epoch_2.pth",
        "./Robo+/Mobile_VLA/original_clip_augmented_results/final_original_clip_augmented.pth",
        "./Robo+/Mobile_VLA/simple_models_original_results/clip_with_lstm/best_clip_with_lstm_epoch_1.pth",
        "./Robo+/Mobile_VLA/simple_models_original_results/simple_clip/best_simple_clip_epoch_2.pth",
        "./Robo+/Mobile_VLA/enhanced_real_data_results/best_enhanced_model_model.pth",
        "./Robo+/Mobile_VLA/original_72_episodes_results/best_original_72_episodes_model_epoch_3.pth",
    ]
    
    print("🔍 모든 모델 체크포인트 분석")
    print("="*80)
    
    results = []
    
    for checkpoint_path in checkpoint_paths:
        info = analyze_checkpoint(checkpoint_path)
        if info:
            results.append(info)
            print(f"✅ {Path(checkpoint_path).name}")
            print(f"   모델 타입: {info['model_type']}")
            print(f"   MAE: {info['mae']}")
            print(f"   Val Loss: {info['val_loss']}")
            print(f"   Train Loss: {info['train_loss']}")
            print(f"   에포크: {info['epoch']}")
            print(f"   파일 크기: {info['file_size_mb']:.2f} MB")
            print()
    
    # MAE 기준 정렬 (숫자만 있는 것들)
    mae_results = []
    for r in results:
        if isinstance(r['mae'], (int, float)):
            mae_results.append(r)
    
    mae_results.sort(key=lambda x: x['mae'])
    
    print("🎯 MAE 기준 성능 순위")
    print("="*60)
    for i, result in enumerate(mae_results, 1):
        print(f"{i}. {Path(result['path']).name}")
        print(f"   모델: {result['model_type']}")
        print(f"   MAE: {result['mae']:.4f}")
        print(f"   Val Loss: {result['val_loss']}")
        print(f"   에포크: {result['epoch']}")
        print()
    
    # 모델별 특징 분석
    print("📊 모델별 특징 분석")
    print("="*60)
    
    model_types = {}
    for result in results:
        model_type = result['model_type']
        if model_type not in model_types:
            model_types[model_type] = []
        model_types[model_type].append(result)
    
    for model_type, models in model_types.items():
        print(f"\n🔹 {model_type} 모델들:")
        for model in models:
            print(f"  - {Path(model['path']).name}: MAE {model['mae']}, Loss {model['val_loss']}")
    
    # 최고 성능 모델 분석
    if mae_results:
        best_model = mae_results[0]
        print(f"\n🏆 최고 성능 모델: {Path(best_model['path']).name}")
        print(f"   MAE: {best_model['mae']:.4f}")
        print(f"   모델 타입: {best_model['model_type']}")
        print(f"   에포크: {best_model['epoch']}")
        print(f"   파일 크기: {best_model['file_size_mb']:.2f} MB")
        
        if 'model_info' in best_model and best_model['model_info']:
            print(f"   모델 정보: {best_model['model_info']}")

if __name__ == "__main__":
    main()
