"""
🔍 2D 액션 모델 성공률 디버깅 스크립트
성공률 계산이 정확한지 확인하고 실제 예측값 분석
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

def debug_accuracy_calculation():
    """성공률 계산 디버깅"""
    print("🔍 2D 액션 모델 성공률 계산 디버깅 시작!")
    
    # 설정
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_path = '../../ROS_action/mobile_vla_dataset'
    batch_size = 4
    
    # 프로세서 로드
    processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
    
    # 모델 로드
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
        print(f"✅ 체크포인트 로드 완료")
    
    model = model.to(device)
    model.eval()
    
    # 소규모 데이터셋으로 테스트
    test_dataset = Optimized2DActionDataset(data_path, processor, 'val')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 성공률 계산을 위한 임계값들
    thresholds = [0.1, 0.05, 0.01]
    
    print(f"\n📊 성공률 계산 디버깅 (임계값: {thresholds})")
    print("="*80)
    
    all_predictions = []
    all_targets = []
    all_errors = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx >= 5:  # 처음 5개 배치만 분석
                break
                
            images = batch['image'].float().to(device)
            actions = batch['action'].float().to(device)
            
            # 예측
            predicted_actions = model(images, "Navigate to target")
            
            # 오차 계산
            errors = torch.abs(predicted_actions - actions)
            
            print(f"\n🔍 배치 {batch_idx + 1}:")
            print(f"   예측값: {predicted_actions.cpu().numpy()}")
            print(f"   타겟값: {actions.cpu().numpy()}")
            print(f"   절대오차: {errors.cpu().numpy()}")
            
            # 각 임계값별 성공률 계산
            for threshold in thresholds:
                # 기존 방식: 모든 차원이 임계값 이내
                all_within = torch.all(errors < threshold, dim=1)
                success_rate_all = (all_within.sum().item() / all_within.shape[0]) * 100
                
                # 개별 차원별 성공률
                linear_x_within = errors[:, 0] < threshold
                linear_y_within = errors[:, 1] < threshold
                
                success_rate_x = (linear_x_within.sum().item() / linear_x_within.shape[0]) * 100
                success_rate_y = (linear_y_within.sum().item() / linear_y_within.shape[0]) * 100
                
                print(f"   임계값 {threshold}:")
                print(f"     전체 성공률: {success_rate_all:.1f}%")
                print(f"     linear_x 성공률: {success_rate_x:.1f}%")
                print(f"     linear_y 성공률: {success_rate_y:.1f}%")
            
            all_predictions.extend(predicted_actions.cpu().numpy())
            all_targets.extend(actions.cpu().numpy())
            all_errors.extend(errors.cpu().numpy())
    
    # 전체 통계
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_errors = np.array(all_errors)
    
    print(f"\n📊 전체 통계:")
    print(f"   총 샘플 수: {len(all_predictions)}")
    print(f"   평균 MAE: {np.mean(all_errors):.4f}")
    print(f"   linear_x 평균 오차: {np.mean(all_errors[:, 0]):.4f}")
    print(f"   linear_y 평균 오차: {np.mean(all_errors[:, 1]):.4f}")
    
    # 각 임계값별 전체 성공률
    print(f"\n🎯 전체 성공률 (새로운 계산):")
    for threshold in thresholds:
        # 개별 차원별 성공률
        linear_x_success = np.mean(all_errors[:, 0] < threshold) * 100
        linear_y_success = np.mean(all_errors[:, 1] < threshold) * 100
        
        # 전체 성공률 (모든 차원이 임계값 이내)
        all_success = np.mean(np.all(all_errors < threshold, axis=1)) * 100
        
        print(f"   임계값 {threshold}:")
        print(f"     전체 성공률: {all_success:.1f}%")
        print(f"     linear_x 성공률: {linear_x_success:.1f}%")
        print(f"     linear_y 성공률: {linear_y_success:.1f}%")
    
    # 성공률 계산 방식 비교
    print(f"\n🔍 성공률 계산 방식 비교:")
    print(f"   기존 방식: 모든 액션 차원이 임계값 이내여야 성공")
    print(f"   개별 차원: 각 차원별로 독립적으로 성공률 계산")
    print(f"   평균 방식: 모든 차원의 평균 오차가 임계값 이내")
    
    # 평균 방식으로 성공률 계산
    print(f"\n📊 평균 방식 성공률:")
    for threshold in thresholds:
        mean_errors = np.mean(all_errors, axis=1)
        mean_success = np.mean(mean_errors < threshold) * 100
        print(f"   임계값 {threshold}: {mean_success:.1f}%")

def analyze_error_distribution():
    """오차 분포 분석"""
    print(f"\n📈 오차 분포 분석:")
    
    # 실제 데이터에서 오차 분포 확인
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_path = '../../ROS_action/mobile_vla_dataset'
    
    # 프로세서 로드
    processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
    
    # 모델 로드
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
        
        # 동적 어댑터 문제 해결
        dummy_input = torch.zeros(1, 3, 224, 224).to(device)
        with torch.no_grad():
            _ = model(dummy_input, "Navigate to target")
        
        model_state_dict = model.state_dict()
        checkpoint_state_dict = checkpoint['model_state_dict']
        
        compatible_state_dict = {}
        for key in checkpoint_state_dict.keys():
            if key in model_state_dict and model_state_dict[key].shape == checkpoint_state_dict[key].shape:
                compatible_state_dict[key] = checkpoint_state_dict[key]
        
        model.load_state_dict(compatible_state_dict, strict=False)
    
    model = model.to(device)
    model.eval()
    
    # 데이터셋 로드
    test_dataset = Optimized2DActionDataset(data_path, processor, 'val')
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    all_errors = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="오차 분포 분석"):
            images = batch['image'].float().to(device)
            actions = batch['action'].float().to(device)
            
            predicted_actions = model(images, "Navigate to target")
            errors = torch.abs(predicted_actions - actions)
            
            all_errors.extend(errors.cpu().numpy())
    
    all_errors = np.array(all_errors)
    
    print(f"   총 샘플 수: {len(all_errors)}")
    print(f"   전체 평균 오차: {np.mean(all_errors):.4f}")
    print(f"   linear_x 평균 오차: {np.mean(all_errors[:, 0]):.4f}")
    print(f"   linear_y 평균 오차: {np.mean(all_errors[:, 1]):.4f}")
    
    # 오차 분포 히스토그램
    print(f"\n📊 오차 분포:")
    for i, dim_name in enumerate(['linear_x', 'linear_y']):
        errors_dim = all_errors[:, i]
        
        # 분위수 계산
        percentiles = [25, 50, 75, 90, 95, 99]
        print(f"   {dim_name} 분위수:")
        for p in percentiles:
            value = np.percentile(errors_dim, p)
            print(f"     {p}%: {value:.4f}")
        
        # 임계값별 성공률
        thresholds = [0.01, 0.05, 0.1, 0.2, 0.5]
        print(f"   {dim_name} 임계값별 성공률:")
        for threshold in thresholds:
            success_rate = np.mean(errors_dim < threshold) * 100
            print(f"     {threshold}: {success_rate:.1f}%")

def main():
    """메인 함수"""
    print("🔍 2D 액션 모델 성공률 디버깅 시작!")
    
    # 성공률 계산 디버깅
    debug_accuracy_calculation()
    
    # 오차 분포 분석
    analyze_error_distribution()
    
    print(f"\n✅ 디버깅 완료!")

if __name__ == "__main__":
    main()
