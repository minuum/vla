#!/usr/bin/env python3
"""
🎯 Z축 특별 처리 전략 - 3D 태스크 완전성 유지
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys

# 프로젝트 경로 설정
ROOT_DIR = Path("/home/billy/25-1kp/vla/Robo+/Mobile_VLA")
DATA_DIR = Path("/home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset")

sys.path.append(str(ROOT_DIR))

def analyze_z_axis_strategies():
    """Z축 처리 전략 분석"""
    print("🎯 Z축 특별 처리 전략 분석")
    print("=" * 50)
    
    print("\n📊 현재 상황:")
    print("   • Z축 데이터: 100% 0값")
    print("   • 3D 태스크: 완전성 필요")
    print("   • 향후 확장: 회전 동작 가능성")
    
    print("\n🔧 처리 전략들:")
    
    # 전략 1: 가중치 조정
    print("\n1️⃣ 가중치 조정 전략:")
    print("   ✅ Z축 가중치를 매우 낮게 설정 (0.05-0.1)")
    print("   ✅ X, Y축은 정상 가중치 (1.0-1.5)")
    print("   ✅ 3D 구조 유지하면서 학습 안정성 확보")
    
    # 전략 2: 정규화 특별 처리
    print("\n2️⃣ 정규화 특별 처리:")
    print("   ✅ Z축 표준편차가 0일 때 기본값 설정 (1.0)")
    print("   ✅ 0으로 나누기 방지")
    print("   ✅ 안전한 정규화 보장")
    
    # 전략 3: 손실 함수 특별 처리
    print("\n3️⃣ 손실 함수 특별 처리:")
    print("   ✅ Z축에 대해 더 관대한 손실 계산")
    print("   ✅ Huber Loss delta 값 조정")
    print("   ✅ NaN 방지 로직 추가")
    
    # 전략 4: 데이터 증강 특별 처리
    print("\n4️⃣ 데이터 증강 특별 처리:")
    print("   ✅ Z축에 매우 작은 노이즈만 추가 (0.001)")
    print("   ✅ X, Y축은 정상적인 증강")
    print("   ✅ 실제 사용 패턴 반영")
    
    # 전략 5: 모델 아키텍처 특별 처리
    print("\n5️⃣ 모델 아키텍처 특별 처리:")
    print("   ✅ Z축 출력에 별도 레이어 추가")
    print("   ✅ Z축 활성화 함수 조정")
    print("   ✅ 그래디언트 흐름 최적화")

def implement_z_axis_strategy():
    """Z축 특별 처리 구현"""
    print("\n🔧 Z축 특별 처리 구현")
    print("=" * 40)
    
    global ZAxisSpecialLoss, ZAxisNormalizer, ZAxisAugmenter
    
    class ZAxisSpecialLoss(nn.Module):
        """Z축 특별 처리 손실 함수"""
        
        def __init__(self, z_weight=0.05, z_delta=0.5):
            super().__init__()
            self.z_weight = z_weight
            self.z_delta = z_delta  # Z축용 더 큰 delta
            
        def forward(self, predictions, targets):
            # 각 축별로 다른 처리
            batch_size, seq_len, action_dim = predictions.shape
            
            # X, Y축: 정상적인 Huber Loss
            xy_loss = F.huber_loss(
                predictions[:, :, :2], 
                targets[:, :, :2], 
                delta=0.1
            )
            
            # Z축: 특별 처리
            z_diff = predictions[:, :, 2] - targets[:, :, 2]
            z_abs_diff = torch.abs(z_diff)
            
            # Z축용 더 관대한 Huber Loss
            z_quadratic = torch.clamp(z_abs_diff, max=self.z_delta)
            z_linear = z_abs_diff - z_quadratic
            z_loss = 0.5 * z_quadratic**2 + self.z_delta * z_linear
            z_loss = z_loss.mean()
            
            # 가중 평균
            total_loss = xy_loss + self.z_weight * z_loss
            
            return total_loss
    
    class ZAxisNormalizer:
        """Z축 특별 정규화"""
        
        def __init__(self):
            self.z_std_fallback = 1.0
            
        def compute_statistics(self, actions):
            """안전한 통계 계산"""
            mean = actions.mean(dim=0)
            std = actions.std(dim=0)
            
            # Z축 특별 처리
            if std[2] < 1e-6:
                print("⚠️ Z축 표준편차가 너무 작음 - 기본값 사용")
                std[2] = self.z_std_fallback
            
            # 안전한 최소값 설정
            std = torch.clamp(std, min=1e-3)
            
            return mean, std
        
        def normalize(self, actions, mean, std):
            """안전한 정규화"""
            normalized = torch.zeros_like(actions)
            
            # X, Y축: 정상 정규화
            for i in range(2):
                normalized[:, :, i] = (actions[:, :, i] - mean[i]) / std[i]
            
            # Z축: 특별 정규화
            if std[2] > 1e-6:
                normalized[:, :, 2] = (actions[:, :, 2] - mean[2]) / std[2]
            else:
                normalized[:, :, 2] = actions[:, :, 2] - mean[2]
            
            return normalized
    
    class ZAxisAugmenter:
        """Z축 특별 증강"""
        
        def augment_actions(self, actions):
            """액션 증강 (Z축 특별 처리)"""
            augmented = actions.clone()
            
            # X, Y축: 정상적인 노이즈
            xy_noise = torch.normal(0, 0.01, actions[:, :, :2].shape)
            augmented[:, :, :2] += xy_noise
            
            # Z축: 매우 작은 노이즈 (실제 사용 패턴 반영)
            z_noise = torch.normal(0, 0.001, actions[:, :, 2:3].shape)
            augmented[:, :, 2:3] += z_noise
            
            # 범위 제한
            augmented = torch.clamp(augmented, -1.15, 1.15)
            
            return augmented
    
    print("✅ Z축 특별 처리 클래스들 구현 완료")
    print("   • ZAxisSpecialLoss: Z축 가중치 조정")
    print("   • ZAxisNormalizer: 안전한 정규화")
    print("   • ZAxisAugmenter: Z축 특별 증강")

def demonstrate_z_axis_handling():
    """Z축 특별 처리 시연"""
    print("\n🎯 Z축 특별 처리 시연")
    print("=" * 40)
    
    # 가상 데이터 생성
    batch_size, seq_len = 2, 8
    actions = torch.zeros(batch_size, seq_len, 3)
    
    # X, Y축: 정상적인 데이터
    actions[:, :, 0] = torch.randn(batch_size, seq_len) * 0.5  # linear_x
    actions[:, :, 1] = torch.randn(batch_size, seq_len) * 0.3  # linear_y
    actions[:, :, 2] = torch.zeros(batch_size, seq_len)        # angular_z (모두 0)
    
    print(f"원본 액션 데이터:")
    print(f"   X축 범위: {actions[:, :, 0].min():.4f} ~ {actions[:, :, 0].max():.4f}")
    print(f"   Y축 범위: {actions[:, :, 1].min():.4f} ~ {actions[:, :, 1].max():.4f}")
    print(f"   Z축 범위: {actions[:, :, 2].min():.4f} ~ {actions[:, :, 2].max():.4f}")
    
    # Z축 특별 처리 적용
    normalizer = ZAxisNormalizer()
    mean, std = normalizer.compute_statistics(actions.view(-1, 3))
    
    print(f"\n정규화 통계:")
    print(f"   평균: {mean}")
    print(f"   표준편차: {std}")
    
    # 정규화 적용
    normalized = normalizer.normalize(actions, mean, std)
    
    print(f"\n정규화 후:")
    print(f"   X축 범위: {normalized[:, :, 0].min():.4f} ~ {normalized[:, :, 0].max():.4f}")
    print(f"   Y축 범위: {normalized[:, :, 1].min():.4f} ~ {normalized[:, :, 1].max():.4f}")
    print(f"   Z축 범위: {normalized[:, :, 2].min():.4f} ~ {normalized[:, :, 2].max():.4f}")
    
    # 데이터 증강 적용
    augmenter = ZAxisAugmenter()
    augmented = augmenter.augment_actions(actions)
    
    print(f"\n증강 후:")
    print(f"   X축 범위: {augmented[:, :, 0].min():.4f} ~ {augmented[:, :, 0].max():.4f}")
    print(f"   Y축 범위: {augmented[:, :, 1].min():.4f} ~ {augmented[:, :, 1].max():.4f}")
    print(f"   Z축 범위: {augmented[:, :, 2].min():.4f} ~ {augmented[:, :, 2].max():.4f}")
    
    # 손실 계산 시연
    predictions = torch.randn_like(actions) * 0.1
    targets = actions
    
    loss_fn = ZAxisSpecialLoss()
    loss = loss_fn(predictions, targets)
    
    print(f"\n특별 손실 계산:")
    print(f"   총 손실: {loss:.6f}")
    print(f"   NaN 여부: {torch.isnan(loss).item()}")
    
    print("\n✅ Z축 특별 처리 시연 완료!")

def future_expansion_plan():
    """향후 확장 계획"""
    print("\n🚀 향후 확장 계획")
    print("=" * 40)
    
    print("📈 단계별 확장:")
    print("1️⃣ 현재: Z축 특별 처리로 안정적 학습")
    print("2️⃣ 중기: 회전 동작 데이터 수집")
    print("3️⃣ 장기: 완전한 3D 로봇 제어")
    
    print("\n🎯 데이터 수집 전략:")
    print("   • 현재: 직진/후진/좌우 이동")
    print("   • 다음: 제자리 회전")
    print("   • 향후: 복합 동작 (이동+회전)")
    
    print("\n🔧 모델 적응 전략:")
    print("   • 현재: Z축 가중치 낮음 (0.05)")
    print("   • 데이터 수집 후: 가중치 점진적 증가")
    print("   • 충분한 데이터 후: 정상 가중치 (1.0)")

if __name__ == "__main__":
    analyze_z_axis_strategies()
    implement_z_axis_strategy()
    demonstrate_z_axis_handling()
    future_expansion_plan()
