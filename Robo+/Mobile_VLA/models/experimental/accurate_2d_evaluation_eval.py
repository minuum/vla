"""
🎯 정확한 2D 액션 모델 평가 스크립트
개별 차원별 성공률과 다양한 계산 방식을 포함한 정확한 성능 평가
"""

import torch
import torch.nn as nn
import numpy as np
import h5py
import os
from pathlib import Path
from transformers import AutoProcessor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import json
from PIL import Image

from optimized_2d_action_model import Optimized2DActionModel, Optimized2DActionDataset

def accurate_2d_evaluation(model, data_loader, device):
    """정확한 2D 액션 모델 평가"""
    model.eval()
    
    total_loss = 0.0
    total_mae = 0.0
    total_rmse = 0.0
    total_samples = 0
    
    # 다양한 성공률 계산을 위한 임계값들
    success_thresholds = [0.01, 0.05, 0.1, 0.2, 0.5]
    
    # 개별 차원별 성공 카운트
    dim_success_counts = {
        'linear_x': {f"accuracy_{int(t*100)}": 0 for t in success_thresholds},
        'linear_y': {f"accuracy_{int(t*100)}": 0 for t in success_thresholds}
    }
    
    # 전체 성공 카운트 (모든 차원 동시)
    all_success_counts = {f"accuracy_{int(t*100)}": 0 for t in success_thresholds}
    
    # 평균 방식 성공 카운트
    mean_success_counts = {f"accuracy_{int(t*100)}": 0 for t in success_thresholds}
    
    # 가중 평균 방식 성공 카운트 (linear_x에 더 높은 가중치)
    weighted_success_counts = {f"accuracy_{int(t*100)}": 0 for t in success_thresholds}
    
    predictions = []
    targets = []
    errors = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="정확한 평가 중"):
            images = batch['image'].float().to(device)
            actions = batch['action'].float().to(device)
            
            # 2D 액션 예측
            predicted_actions = model(images, "Navigate to target")
            
            # 손실 계산
            loss = nn.functional.mse_loss(predicted_actions, actions)
            total_loss += loss.item()
            
            # MAE 계산
            mae = nn.functional.l1_loss(predicted_actions, actions)
            total_mae += mae.item()
            
            # RMSE 계산
            rmse = torch.sqrt(nn.functional.mse_loss(predicted_actions, actions))
            total_rmse += rmse.item()
            
            # 오차 계산
            batch_errors = torch.abs(predicted_actions - actions)
            
            # 성공률 계산 (다양한 방식)
            for threshold in success_thresholds:
                accuracy_key = f"accuracy_{int(threshold*100)}"
                
                # 1. 개별 차원별 성공률
                linear_x_within = batch_errors[:, 0] < threshold
                linear_y_within = batch_errors[:, 1] < threshold
                
                dim_success_counts['linear_x'][accuracy_key] += linear_x_within.sum().item()
                dim_success_counts['linear_y'][accuracy_key] += linear_y_within.sum().item()
                
                # 2. 전체 성공률 (모든 차원 동시)
                all_within = torch.all(batch_errors < threshold, dim=1)
                all_success_counts[accuracy_key] += all_within.sum().item()
                
                # 3. 평균 방식 성공률
                mean_errors = torch.mean(batch_errors, dim=1)
                mean_within = mean_errors < threshold
                mean_success_counts[accuracy_key] += mean_within.sum().item()
                
                # 4. 가중 평균 방식 성공률 (linear_x: 0.7, linear_y: 0.3)
                weighted_errors = 0.7 * batch_errors[:, 0] + 0.3 * batch_errors[:, 1]
                weighted_within = weighted_errors < threshold
                weighted_success_counts[accuracy_key] += weighted_within.sum().item()
            
            total_samples += images.shape[0]
            
            # 예측값과 타겟 저장
            predictions.extend(predicted_actions.cpu().numpy())
            targets.extend(actions.cpu().numpy())
            errors.extend(batch_errors.cpu().numpy())
    
    # 평균 계산
    avg_loss = total_loss / len(data_loader)
    avg_mae = total_mae / len(data_loader)
    avg_rmse = total_rmse / len(data_loader)
    
    # 성공률 계산
    success_rates = {}
    
    # 개별 차원별 성공률
    for dim_name, counts in dim_success_counts.items():
        success_rates[f"{dim_name}_success_rates"] = {}
        for key, count in counts.items():
            success_rates[f"{dim_name}_success_rates"][key] = (count / total_samples) * 100
    
    # 전체 성공률 (모든 차원 동시)
    success_rates['all_dimensions_success_rates'] = {}
    for key, count in all_success_counts.items():
        success_rates['all_dimensions_success_rates'][key] = (count / total_samples) * 100
    
    # 평균 방식 성공률
    success_rates['mean_success_rates'] = {}
    for key, count in mean_success_counts.items():
        success_rates['mean_success_rates'][key] = (count / total_samples) * 100
    
    # 가중 평균 방식 성공률
    success_rates['weighted_success_rates'] = {}
    for key, count in weighted_success_counts.items():
        success_rates['weighted_success_rates'][key] = (count / total_samples) * 100
    
    # 예측값과 타겟을 numpy 배열로 변환
    predictions = np.array(predictions)
    targets = np.array(targets)
    errors = np.array(errors)
    
    # 각 액션 차원별 상세 성능 분석
    action_dim_performance = {}
    for i, dim_name in enumerate(['linear_x', 'linear_y']):
        dim_predictions = predictions[:, i]
        dim_targets = targets[:, i]
        dim_errors = errors[:, i]
        
        dim_mae = np.mean(dim_errors)
        dim_rmse = np.sqrt(np.mean(dim_errors ** 2))
        
        # 분위수 계산
        percentiles = [25, 50, 75, 90, 95, 99]
        percentile_values = {}
        for p in percentiles:
            percentile_values[f"{p}th_percentile"] = float(np.percentile(dim_errors, p))
        
        action_dim_performance[dim_name] = {
            'mae': float(dim_mae),
            'rmse': float(dim_rmse),
            'percentiles': percentile_values,
            'min_error': float(np.min(dim_errors)),
            'max_error': float(np.max(dim_errors)),
            'std_error': float(np.std(dim_errors))
        }
    
    results = {
        'model_type': 'Optimized_2D_Action_Model_Accurate',
        'total_samples': total_samples,
        'avg_loss': float(avg_loss),
        'avg_mae': float(avg_mae),
        'avg_rmse': float(avg_rmse),
        'success_rates': success_rates,
        'action_dim_performance': action_dim_performance,
        'predictions_shape': predictions.shape,
        'targets_shape': targets.shape,
        'evaluation_methods': {
            'individual_dimensions': '각 차원별 독립적 성공률',
            'all_dimensions': '모든 차원이 동시에 임계값 이내',
            'mean_errors': '두 차원의 평균 오차가 임계값 이내',
            'weighted_mean': '가중 평균 오차 (linear_x: 0.7, linear_y: 0.3)'
        }
    }
    
    return results

def print_accurate_results(results):
    """정확한 결과 출력"""
    print("\n" + "="*80)
    print("🎯 정확한 2D 액션 모델 평가 결과")
    print("="*80)
    print(f"📊 총 샘플 수: {results['total_samples']:,}")
    print(f"📊 평균 손실: {results['avg_loss']:.4f}")
    print(f"📊 평균 MAE: {results['avg_mae']:.4f}")
    print(f"📊 평균 RMSE: {results['avg_rmse']:.4f}")
    
    print("\n🎯 성공률 비교 (다양한 계산 방식):")
    print("-" * 80)
    
    thresholds = [0.01, 0.05, 0.1, 0.2, 0.5]
    
    # 헤더
    print(f"{'임계값':<8} {'Linear_X':<10} {'Linear_Y':<10} {'전체(동시)':<12} {'평균':<8} {'가중평균':<10}")
    print("-" * 80)
    
    for threshold in thresholds:
        acc_key = f"accuracy_{int(threshold*100)}"
        
        linear_x_rate = results['success_rates']['linear_x_success_rates'][acc_key]
        linear_y_rate = results['success_rates']['linear_y_success_rates'][acc_key]
        all_rate = results['success_rates']['all_dimensions_success_rates'][acc_key]
        mean_rate = results['success_rates']['mean_success_rates'][acc_key]
        weighted_rate = results['success_rates']['weighted_success_rates'][acc_key]
        
        print(f"{threshold:<8} {linear_x_rate:<10.1f}% {linear_y_rate:<10.1f}% {all_rate:<12.1f}% {mean_rate:<8.1f}% {weighted_rate:<10.1f}%")
    
    print("\n📊 액션 차원별 상세 성능:")
    print("-" * 80)
    for dim_name, performance in results['action_dim_performance'].items():
        print(f"🔍 {dim_name}:")
        print(f"   MAE: {performance['mae']:.4f}")
        print(f"   RMSE: {performance['rmse']:.4f}")
        print(f"   표준편차: {performance['std_error']:.4f}")
        print(f"   최소 오차: {performance['min_error']:.4f}")
        print(f"   최대 오차: {performance['max_error']:.4f}")
        print(f"   분위수:")
        for p_key, p_value in performance['percentiles'].items():
            print(f"     {p_key}: {p_value:.4f}")
    
    print("\n💡 평가 방식 설명:")
    for method, description in results['evaluation_methods'].items():
        print(f"   - {method}: {description}")

def main():
    """메인 평가 함수"""
    print("🎯 정확한 2D 액션 모델 평가 시작!")
    
    # 설정
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_path = '../../ROS_action/mobile_vla_dataset'
    batch_size = 8
    
    print(f"📊 설정: 디바이스={device}, 배치크기={batch_size}")
    
    # 프로세서 로드
    print("🔧 프로세서 로드 중...")
    processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
    
    # 평가 데이터셋 로드
    print("📊 평가 데이터셋 로드 중...")
    eval_dataset = Optimized2DActionDataset(data_path, processor, 'val')
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    
    # 모델 로드
    print("🤖 2D 액션 최적화 모델 로드 중...")
    model = Optimized2DActionModel(
        processor=processor,
        action_dim=2,
        dropout=0.2,
        use_claw_matrix=True,
        use_hierarchical=True,
        use_advanced_attention=True
    )
    
    # 체크포인트 로드
    checkpoint_path = 'optimized_2d_action_model_best.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 동적 어댑터 문제 해결을 위한 더미 포워드 패스
        dummy_input = torch.zeros(1, 3, 224, 224).to(device)
        with torch.no_grad():
            _ = model(dummy_input, "Navigate to target")
        
        # 호환되는 키만 로드
        model_state_dict = model.state_dict()
        checkpoint_state_dict = checkpoint['model_state_dict']
        
        compatible_state_dict = {}
        for key in checkpoint_state_dict.keys():
            if key in model_state_dict and model_state_dict[key].shape == checkpoint_state_dict[key].shape:
                compatible_state_dict[key] = checkpoint_state_dict[key]
        
        model.load_state_dict(compatible_state_dict, strict=False)
        print(f"✅ 체크포인트 로드 완료: {checkpoint_path}")
        print(f"   - 에포크: {checkpoint['epoch']}")
        print(f"   - 검증 손실: {checkpoint['val_loss']:.4f}")
    else:
        print(f"❌ 체크포인트를 찾을 수 없습니다: {checkpoint_path}")
        return
    
    model = model.to(device)
    
    # 정확한 모델 평가
    print("🎯 정확한 모델 평가 시작...")
    results = accurate_2d_evaluation(model, eval_loader, device)
    
    # 결과 출력
    print_accurate_results(results)
    
    # 결과 저장
    with open('accurate_2d_action_evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 정확한 평가 결과 저장 완료: accurate_2d_action_evaluation_results.json")
    
    # 최종 요약
    print("\n" + "="*80)
    print("🎯 최종 요약")
    print("="*80)
    print(f"✅ 정확한 2D 액션 모델 평가 완료!")
    print(f"📊 주요 성능 지표:")
    print(f"   - 평균 MAE: {results['avg_mae']:.4f}")
    print(f"   - 평균 RMSE: {results['avg_rmse']:.4f}")
    print(f"   - Linear_X 성공률 (0.1): {results['success_rates']['linear_x_success_rates']['accuracy_10']:.1f}%")
    print(f"   - Linear_Y 성공률 (0.1): {results['success_rates']['linear_y_success_rates']['accuracy_10']:.1f}%")
    print(f"   - 가중 평균 성공률 (0.1): {results['success_rates']['weighted_success_rates']['accuracy_10']:.1f}%")

if __name__ == "__main__":
    main()
