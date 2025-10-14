#!/usr/bin/env python3
"""
🎯 최종 성능 추출 및 비교표 생성
기존 모델들의 정확한 성능 지표 추출
"""

import json
import os
from pathlib import Path
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_performance_from_history(history_data):
    """학습 히스토리에서 최고 성능 추출"""
    if isinstance(history_data, list):
        # 리스트 형태의 히스토리
        best_mae = min([epoch.get('val_mae', float('inf')) for epoch in history_data])
        best_val_loss = min([epoch.get('val_loss', float('inf')) for epoch in history_data])
        final_train_mae = history_data[-1].get('train_mae', 'N/A')
        final_train_loss = history_data[-1].get('train_loss', 'N/A')
        epochs = len(history_data)
    elif isinstance(history_data, dict):
        # 딕셔너리 형태의 히스토리
        if 'val_mae' in history_data:
            # 단일 값들
            best_mae = min(history_data['val_mae'])
            best_val_loss = min(history_data['val_loss'])
            final_train_mae = history_data['train_mae'][-1]
            final_train_loss = history_data['train_loss'][-1]
            epochs = len(history_data['val_mae'])
        elif 'training_history' in history_data:
            # 중첩된 히스토리
            training_history = history_data['training_history']
            best_mae = min([epoch.get('val_mae', float('inf')) for epoch in training_history])
            best_val_loss = min([epoch.get('val_loss', float('inf')) for epoch in training_history])
            final_train_mae = training_history[-1].get('train_mae', 'N/A')
            final_train_loss = training_history[-1].get('train_loss', 'N/A')
            epochs = len(training_history)
        elif 'val_maes' in history_data:
            # 다른 형태의 히스토리
            best_mae = min(history_data['val_maes'])
            best_val_loss = min(history_data['val_losses'])
            final_train_mae = history_data['train_losses'][-1]
            final_train_loss = history_data['train_losses'][-1]
            epochs = history_data.get('final_epoch', len(history_data['val_maes']))
        else:
            best_mae = 'N/A'
            best_val_loss = 'N/A'
            final_train_mae = 'N/A'
            final_train_loss = 'N/A'
            epochs = 'N/A'
    else:
        best_mae = 'N/A'
        best_val_loss = 'N/A'
        final_train_mae = 'N/A'
        final_train_loss = 'N/A'
        epochs = 'N/A'
    
    return {
        'best_mae': best_mae,
        'best_val_loss': best_val_loss,
        'final_train_mae': final_train_mae,
        'final_train_loss': final_train_loss,
        'epochs': epochs
    }

def main():
    """메인 함수"""
    logger.info("🔍 최종 성능 추출 시작")
    
    # 기존 분석 결과 로드
    with open('action_head_analysis_results.json', 'r') as f:
        analysis_data = json.load(f)
    
    # 성능 데이터 추출
    performance_data = []
    
    for model_analysis in analysis_data['model_analyses']:
        model_info = {
            'path': model_analysis['path'],
            'action_head_type': model_analysis['action_head_type'],
            'action_dim': model_analysis['action_dim'],
            'model_size_mb': model_analysis['model_size_mb'],
            'epoch': model_analysis['epoch']
        }
        
        # 학습 히스토리에서 성능 추출
        if 'training_history' in model_analysis:
            performance = extract_performance_from_history(model_analysis['training_history'])
            model_info.update(performance)
        else:
            model_info.update({
                'best_mae': 'N/A',
                'best_val_loss': 'N/A',
                'final_train_mae': 'N/A',
                'final_train_loss': 'N/A',
                'epochs': 'N/A'
            })
        
        performance_data.append(model_info)
    
    # Action Head 타입별 그룹화
    lstm_models = [m for m in performance_data if m['action_head_type'] == 'LSTM']
    mlp_models = [m for m in performance_data if m['action_head_type'] == 'MLP']
    
    # MAE 기준으로 정렬 (유효한 값만)
    def sort_by_mae(models):
        valid_models = [m for m in models if m['best_mae'] != 'N/A' and isinstance(m['best_mae'], (int, float))]
        return sorted(valid_models, key=lambda x: x['best_mae'])
    
    lstm_models_sorted = sort_by_mae(lstm_models)
    mlp_models_sorted = sort_by_mae(mlp_models)
    
    # 결과 출력
    print("\n" + "="*100)
    print("🎯 VLM + Action Head 구조 모델 최종 성능 비교표")
    print("="*100)
    
    print(f"\n🥇 **LSTM Action Head 모델들 (MAE 기준 정렬):**")
    print("-" * 80)
    print(f"{'순위':<4} {'모델명':<50} {'MAE':<8} {'Val Loss':<10} {'Train MAE':<10} {'에포크':<6} {'크기(GB)':<8}")
    print("-" * 80)
    
    for i, model in enumerate(lstm_models_sorted, 1):
        model_name = Path(model['path']).name
        mae = f"{model['best_mae']:.4f}" if model['best_mae'] != 'N/A' else 'N/A'
        val_loss = f"{model['best_val_loss']:.4f}" if model['best_val_loss'] != 'N/A' else 'N/A'
        train_mae = f"{model['final_train_mae']:.4f}" if model['final_train_mae'] != 'N/A' else 'N/A'
        epochs = model['epochs'] if model['epochs'] != 'N/A' else 'N/A'
        size_gb = f"{model['model_size_mb']/1024:.2f}"
        
        print(f"{i:<4} {model_name:<50} {mae:<8} {val_loss:<10} {train_mae:<10} {epochs:<6} {size_gb:<8}")
    
    print(f"\n🥈 **MLP Action Head 모델들 (MAE 기준 정렬):**")
    print("-" * 80)
    print(f"{'순위':<4} {'모델명':<50} {'MAE':<8} {'Val Loss':<10} {'Train MAE':<10} {'에포크':<6} {'크기(GB)':<8}")
    print("-" * 80)
    
    for i, model in enumerate(mlp_models_sorted, 1):
        model_name = Path(model['path']).name
        mae = f"{model['best_mae']:.4f}" if model['best_mae'] != 'N/A' else 'N/A'
        val_loss = f"{model['best_val_loss']:.4f}" if model['best_val_loss'] != 'N/A' else 'N/A'
        train_mae = f"{model['final_train_mae']:.4f}" if model['final_train_mae'] != 'N/A' else 'N/A'
        epochs = model['epochs'] if model['epochs'] != 'N/A' else 'N/A'
        size_gb = f"{model['model_size_mb']/1024:.2f}"
        
        print(f"{i:<4} {model_name:<50} {mae:<8} {val_loss:<10} {train_mae:<10} {epochs:<6} {size_gb:<8}")
    
    # Action Head 타입별 최고 성능
    print(f"\n🏆 **Action Head 타입별 최고 성능:**")
    print("-" * 60)
    
    if lstm_models_sorted:
        best_lstm = lstm_models_sorted[0]
        print(f"LSTM Action Head: MAE {best_lstm['best_mae']:.4f} ({Path(best_lstm['path']).name})")
    
    if mlp_models_sorted:
        best_mlp = mlp_models_sorted[0]
        print(f"MLP Action Head:  MAE {best_mlp['best_mae']:.4f} ({Path(best_mlp['path']).name})")
    
    # 종합 순위
    all_models_sorted = sort_by_mae(performance_data)
    print(f"\n🥇 **전체 모델 종합 순위 (상위 5개):**")
    print("-" * 80)
    print(f"{'순위':<4} {'Action Head':<12} {'모델명':<50} {'MAE':<8}")
    print("-" * 80)
    
    for i, model in enumerate(all_models_sorted[:5], 1):
        model_name = Path(model['path']).name
        mae = f"{model['best_mae']:.4f}"
        action_head = model['action_head_type']
        
        print(f"{i:<4} {action_head:<12} {model_name:<50} {mae:<8}")
    
    # 결과를 JSON으로 저장
    final_results = {
        'lstm_models': lstm_models_sorted,
        'mlp_models': mlp_models_sorted,
        'all_models_ranked': all_models_sorted,
        'best_lstm': lstm_models_sorted[0] if lstm_models_sorted else None,
        'best_mlp': mlp_models_sorted[0] if mlp_models_sorted else None,
        'overall_best': all_models_sorted[0] if all_models_sorted else None
    }
    
    with open('final_performance_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    logger.info("최종 성능 추출 완료!")
    logger.info("결과가 final_performance_results.json에 저장되었습니다.")

if __name__ == "__main__":
    main()
