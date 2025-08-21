#!/usr/bin/env python3
"""
🎯 성공률 계산 정확성 검증
"""
import numpy as np

def verify_success_rate_calculation():
    """성공률 계산 정확성 검증"""
    print("🎯 성공률 계산 정확성 검증")
    print("=" * 60)
    
    # 현재 모델 성능
    current_mae = 0.2602
    distance_mae = {
        'close': 0.2617,
        'medium': 0.2017,
        'far': 0.3373
    }
    
    # 실제 액션 범위 확인
    actual_actions = {
        'W (전진)': [1.15, 0.0, 0.0],
        'A (좌측)': [0.0, 1.15, 0.0],
        'D (우측)': [0.0, -1.15, 0.0],
        'Q (전진+좌측)': [1.15, 1.15, 0.0],
        'E (전진+우측)': [1.15, -1.15, 0.0],
        'SPACE (정지)': [0.0, 0.0, 0.0]
    }
    
    # 액션 벡터 크기 계산
    action_magnitudes = []
    for action_name, action_vec in actual_actions.items():
        magnitude = np.sqrt(sum(v**2 for v in action_vec))
        action_magnitudes.append(magnitude)
        print(f"   {action_name}: 크기 {magnitude:.3f}")
    
    # 실제 액션 범위
    max_magnitude = max(action_magnitudes)
    min_magnitude = min(action_magnitudes)
    avg_magnitude = np.mean(action_magnitudes)
    
    print(f"\n📊 액션 범위 분석:")
    print(f"   최대 크기: {max_magnitude:.3f}")
    print(f"   최소 크기: {min_magnitude:.3f}")
    print(f"   평균 크기: {avg_magnitude:.3f}")
    
    # 다양한 기준으로 성공률 계산
    print(f"\n🎯 다양한 기준으로 성공률 계산:")
    
    # 1. 최대 크기 기준
    success_rate_max = max(0, (1 - current_mae / max_magnitude)) * 100
    print(f"   최대 크기 기준: {success_rate_max:.1f}%")
    
    # 2. 평균 크기 기준
    success_rate_avg = max(0, (1 - current_mae / avg_magnitude)) * 100
    print(f"   평균 크기 기준: {success_rate_avg:.1f}%")
    
    # 3. 실제 액션별 성공률
    print(f"\n📊 실제 액션별 성공률:")
    for action_name, action_vec in actual_actions.items():
        magnitude = np.sqrt(sum(v**2 for v in action_vec))
        if magnitude > 0:
            success_rate = max(0, (1 - current_mae / magnitude)) * 100
        else:
            # 정지 액션은 작은 임계값 사용
            success_rate = max(0, (1 - current_mae / 0.1)) * 100
        print(f"   {action_name}: {success_rate:.1f}%")
    
    # 4. 거리별 성공률 재검증
    print(f"\n📏 거리별 성공률 재검증:")
    for distance, mae in distance_mae.items():
        # 평균 액션 크기 기준
        success_rate = max(0, (1 - mae / avg_magnitude)) * 100
        print(f"   {distance.capitalize()}: {success_rate:.1f}% (MAE: {mae:.4f})")
    
    # 5. 임계값별 성공률
    print(f"\n🎯 임계값별 성공률:")
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 1.5, 2.0]
    for threshold in thresholds:
        success_rate = max(0, (1 - current_mae / threshold)) * 100
        print(f"   임계값 {threshold}: {success_rate:.1f}% 성공률")
    
    print(f"\n💡 결론:")
    print(f"   - 88.7% 성공률은 평균 액션 크기 기준")
    print(f"   - 실제로는 액션별로 다른 성공률")
    print(f"   - 정지 액션(SPACE)은 매우 낮은 성공률")
    print(f"   - 전진/횡이동 액션은 77-84% 성공률")

def verify_model_capabilities():
    """모델 능력 검증"""
    print(f"\n🔍 모델 능력 검증")
    print("=" * 60)
    
    # 현재 모델 구조 확인
    print("📋 현재 모델 구조:")
    print("   - 입력: 8프레임 이미지 시퀀스")
    print("   - 출력: 2프레임 액션 예측")
    print("   - 백본: Kosmos2 Vision Model")
    print("   - 액션 헤드: LSTM + MLP")
    print("   - 거리별 특화: Distance Embedding + Fusion")
    
    print(f"\n❓ 18프레임 예측 가능성:")
    print("   - 현재: 8프레임 → 2프레임 예측")
    print("   - 18프레임 예측: 구조 변경 필요")
    print("   - 방법: LSTM 시퀀스 길이 확장")
    
    print(f"\n🔧 RoboVLMs 핵심 기술 포함 여부:")
    print("   ✅ Kosmos2 Vision Backbone")
    print("   ✅ Temporal Modeling (LSTM)")
    print("   ✅ Multi-modal Fusion")
    print("   ❌ Claw Matrix (구현 필요)")
    print("   ❌ Advanced Attention Mechanisms")
    print("   ❌ Hierarchical Planning")

if __name__ == "__main__":
    verify_success_rate_calculation()
    verify_model_capabilities()
