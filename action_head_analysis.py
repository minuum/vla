#!/usr/bin/env python3
"""
🎯 VLM + Action Head 구조 모델 분석
각 모델의 Action Head 타입별 성능 비교
"""

import torch
import torch.nn as nn
import json
import os
from pathlib import Path
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_model_action_head(model_path: str) -> dict:
    """모델의 Action Head 구조 분석"""
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # 모델 구조 분석
        model_info = {
            'path': model_path,
            'action_head_type': 'Unknown',
            'action_dim': 'Unknown',
            'has_lstm': False,
            'has_mlp': False,
            'has_gpt2': False,
            'has_discrete': False,
            'model_size_mb': os.path.getsize(model_path) / (1024 * 1024),
            'mae': 'N/A',
            'val_loss': 'N/A',
            'train_loss': 'N/A',
            'epoch': 'N/A'
        }
        
        # 체크포인트에서 모델 구조 추출
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Action Head 타입 분석
        action_head_keys = [k for k in state_dict.keys() if 'action_head' in k.lower()]
        lstm_keys = [k for k in state_dict.keys() if 'lstm' in k.lower()]
        mlp_keys = [k for k in state_dict.keys() if any(x in k.lower() for x in ['linear', 'fc', 'dense'])]
        gpt2_keys = [k for k in state_dict.keys() if 'gpt' in k.lower() or 'transformer' in k.lower()]
        discrete_keys = [k for k in state_dict.keys() if 'discrete' in k.lower() or 'token' in k.lower()]
        
        # Action Head 타입 결정
        if lstm_keys and any('action_head' in k for k in action_head_keys):
            model_info['action_head_type'] = 'LSTM'
            model_info['has_lstm'] = True
        elif mlp_keys and any('action_head' in k for k in action_head_keys):
            model_info['action_head_type'] = 'MLP'
            model_info['has_mlp'] = True
        elif gpt2_keys:
            model_info['action_head_type'] = 'GPT2'
            model_info['has_gpt2'] = True
        elif discrete_keys:
            model_info['action_head_type'] = 'Discrete'
            model_info['has_discrete'] = True
        else:
            # 기본적으로 LSTM으로 분류 (대부분의 모델이 LSTM 사용)
            if lstm_keys:
                model_info['action_head_type'] = 'LSTM'
                model_info['has_lstm'] = True
            elif mlp_keys:
                model_info['action_head_type'] = 'MLP'
                model_info['has_mlp'] = True
        
        # Action 차원 추출
        for key in action_head_keys:
            if 'weight' in key and len(state_dict[key].shape) == 2:
                # 마지막 레이어의 출력 차원
                if state_dict[key].shape[0] in [2, 3, 4]:  # 일반적인 액션 차원
                    model_info['action_dim'] = state_dict[key].shape[0]
                    break
        
        # 성능 지표 추출
        if 'mae' in checkpoint:
            model_info['mae'] = checkpoint['mae']
        if 'val_loss' in checkpoint:
            model_info['val_loss'] = checkpoint['val_loss']
        if 'train_loss' in checkpoint:
            model_info['train_loss'] = checkpoint['train_loss']
        if 'epoch' in checkpoint:
            model_info['epoch'] = checkpoint['epoch']
        
        return model_info
        
    except Exception as e:
        logger.error(f"Error analyzing {model_path}: {e}")
        return {
            'path': model_path,
            'action_head_type': 'Error',
            'error': str(e)
        }

def find_training_history(model_path: str) -> dict:
    """학습 히스토리 파일 찾기"""
    model_dir = Path(model_path).parent
    history_files = list(model_dir.glob("*history*.json")) + list(model_dir.glob("*training*.json"))
    
    for history_file in history_files:
        try:
            with open(history_file, 'r') as f:
                history = json.load(f)
                return history
        except:
            continue
    
    return {}

def main():
    """메인 분석 함수"""
    logger.info("🔍 VLM + Action Head 구조 모델 분석 시작")
    
    # 모델 경로들
    model_paths = [
        # Enhanced Kosmos2+CLIP 모델들
        "enhanced_kosmos2_clip_hybrid_results/best_enhanced_kosmos2_clip_hybrid.pth",
        "enhanced_kosmos2_clip_hybrid_with_normalization_results/best_enhanced_kosmos2_clip_hybrid_with_mobile_normalization.pth",
        
        # 기존 모델들
        "best_simple_clip_lstm_model.pth",
        "final_simple_lstm_model.pth",
        "best_model_epoch_3.pt",
        "best_model_epoch_2.pt",
        "best_model_epoch_1.pt",
        
        # Mobile VLA 모델들
        "Robo+/Mobile_VLA/results/mobile_vla_epoch_3.pt",
        "Robo+/Mobile_VLA/results/mobile_vla_epoch_2.pt",
        "Robo+/Mobile_VLA/results/mobile_vla_epoch_1.pt",
        
        # Simple 모델들
        "Robo+/Mobile_VLA/simple_models_original_results/simple_clip/best_simple_clip_epoch_2.pth",
        "Robo+/Mobile_VLA/simple_models_original_results/clip_with_lstm/best_clip_with_lstm_epoch_1.pth",
        
        # Original 모델들
        "Robo+/Mobile_VLA/original_clip_augmented_results/best_original_clip_augmented_epoch_2.pth",
        "best_original_72_episodes_model_epoch_3.pth"
    ]
    
    # 모델 분석 결과
    model_analyses = []
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            logger.info(f"분석 중: {model_path}")
            analysis = analyze_model_action_head(model_path)
            
            # 학습 히스토리 추가
            history = find_training_history(model_path)
            if history:
                analysis['training_history'] = history
            
            model_analyses.append(analysis)
        else:
            logger.warning(f"모델 파일을 찾을 수 없음: {model_path}")
    
    # Action Head 타입별 그룹화
    action_head_groups = {}
    for analysis in model_analyses:
        head_type = analysis['action_head_type']
        if head_type not in action_head_groups:
            action_head_groups[head_type] = []
        action_head_groups[head_type].append(analysis)
    
    # 결과 출력
    print("\n" + "="*80)
    print("🎯 VLM + Action Head 구조 모델 분석 결과")
    print("="*80)
    
    for head_type, models in action_head_groups.items():
        print(f"\n📊 {head_type} Action Head 모델들:")
        print("-" * 50)
        
        # MAE 기준으로 정렬
        valid_models = [m for m in models if m['mae'] != 'N/A' and isinstance(m['mae'], (int, float))]
        valid_models.sort(key=lambda x: x['mae'])
        
        for i, model in enumerate(valid_models, 1):
            print(f"{i:2d}. {Path(model['path']).name}")
            print(f"    MAE: {model['mae']:.4f}")
            print(f"    Val Loss: {model['val_loss']}")
            print(f"    Action Dim: {model['action_dim']}")
            print(f"    Model Size: {model['model_size_mb']:.1f} MB")
            print(f"    Epoch: {model['epoch']}")
            print()
    
    # Action Head 타입별 성능 비교
    print("\n🏆 Action Head 타입별 최고 성능:")
    print("-" * 50)
    
    for head_type, models in action_head_groups.items():
        if head_type == 'Error':
            continue
            
        valid_models = [m for m in models if m['mae'] != 'N/A' and isinstance(m['mae'], (int, float))]
        if valid_models:
            best_model = min(valid_models, key=lambda x: x['mae'])
            print(f"{head_type:10s}: MAE {best_model['mae']:.4f} ({Path(best_model['path']).name})")
    
    # 결과를 JSON으로 저장
    output_file = "action_head_analysis_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'model_analyses': model_analyses,
            'action_head_groups': action_head_groups
        }, f, indent=2)
    
    logger.info(f"분석 결과가 {output_file}에 저장되었습니다.")

if __name__ == "__main__":
    main()
