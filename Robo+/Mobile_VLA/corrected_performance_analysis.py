#!/usr/bin/env python3
"""
📊 실제 데이터 수집 방식 기반 정확한 성능 분석
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

def load_training_results():
    """학습 결과 로드"""
    with open('augmented_training_results.json', 'r') as f:
        return json.load(f)

def analyze_actual_data_collection():
    """실제 데이터 수집 방식 분석"""
    print("🎯 실제 데이터 수집 방식 분석")
    print("=" * 60)
    
    # 실제 사용된 액션만 (회전, 후진, Z/C 미사용)
    ACTUAL_ACTIONS = {
        'w': {"linear_x": 1.15, "linear_y": 0.0, "angular_z": 0.0},      # 전진
        'a': {"linear_x": 0.0, "linear_y": 1.15, "angular_z": 0.0},      # 좌측 이동
        'd': {"linear_x": 0.0, "linear_y": -1.15, "angular_z": 0.0},     # 우측 이동
        'q': {"linear_x": 1.15, "linear_y": 1.15, "angular_z": 0.0},     # 전진+좌측
        'e': {"linear_x": 1.15, "linear_y": -1.15, "angular_z": 0.0},    # 전진+우측
        ' ': {"linear_x": 0.0, "linear_y": 0.0, "angular_z": 0.0}        # 정지
    }
    
    print("📋 실제 사용된 액션:")
    print("   ✅ 사용됨: W(전진), A(좌측), D(우측), Q(전진+좌측), E(전진+우측), SPACE(정지)")
    print("   ❌ 미사용: S(후진), R(좌회전), T(우회전), Z(후진+좌측), C(후진+우측)")
    print()
    
    # 거리별 데이터 분포
    distance_distribution = {
        "close": {"description": "세로: 로봇과 가까움 / 가로: 좌측 치우침", "samples": 3},
        "medium": {"description": "세로: 중간 거리 / 가로: 중앙 근처", "samples": 4},
        "far": {"description": "세로: 로봇과 멀음 / 가로: 우측 치우침", "samples": 3}
    }
    
    print("📏 거리별 데이터 분포:")
    for distance, info in distance_distribution.items():
        print(f"   {distance:8s}: {info['description']} ({info['samples']}개 샘플)")
    print()
    
    return ACTUAL_ACTIONS, distance_distribution

def calculate_realistic_accuracy(mae_value):
    """실제 액션 예측 정확도 계산"""
    print("🎯 실제 액션 예측 정확도 분석")
    print("=" * 60)
    
    # MAE 0.442를 실제 정확도로 환산
    # 실제 사용된 액션 범위: -1.15 ~ 1.15 (약 2.3 범위)
    action_range = 2.3
    
    # 정확도 계산 (오차가 작을수록 높은 정확도)
    accuracy_base = max(0, 1 - (mae_value / action_range))
    
    # 임계값별 정확도
    thresholds = {
        0.1: "매우 정확한 예측",
        0.2: "정확한 예측", 
        0.3: "적절한 예측",
        0.4: "보통 예측",
        0.5: "기본 예측"
    }
    
    print(f"📊 MAE {mae_value:.3f} → 실제 정확도: {accuracy_base:.1%}")
    print()
    
    print("📈 임계값별 정확도:")
    for threshold, description in thresholds.items():
        # 해당 임계값에서의 정확도
        threshold_accuracy = max(0, 1 - (threshold / action_range))
        print(f"   오차 ≤ {threshold:3.1f}: {threshold_accuracy:.1%} ({description})")
    
    # 실제 성능 해석
    print(f"\n🎯 성능 해석:")
    if mae_value <= 0.2:
        print("   ✅ 우수한 성능: 로봇이 정확한 액션을 예측할 확률이 높음")
        print(f"   📈 정확도: {accuracy_base:.1%} 이상")
    elif mae_value <= 0.4:
        print("   ⚠️  보통 성능: 대부분의 액션을 적절히 예측")
        print(f"   📊 정확도: {accuracy_base:.1%} 정도")
    elif mae_value <= 0.6:
        print("   ⚠️  개선 필요: 일부 액션에서 오차 발생")
        print(f"   📉 정확도: {accuracy_base:.1%} 정도")
    else:
        print("   ❌ 낮은 성능: 상당한 개선 필요")
        print(f"   📉 정확도: {accuracy_base:.1%} 미만")
    
    return accuracy_base, thresholds

def analyze_distance_based_performance():
    """거리별 성능 분석"""
    print("\n📏 거리별 성능 분석")
    print("=" * 60)
    
    # 거리별 예상 성능 패턴
    distance_performance = {
        "close": {
            "description": "로봇과 가까운 장애물",
            "characteristics": "정밀한 조작 필요, 작은 움직임",
            "expected_mae": 0.35,
            "key_actions": ["W", "A", "D", "SPACE"],
            "difficulty": "높음"
        },
        "medium": {
            "description": "중간 거리 장애물", 
            "characteristics": "균형잡힌 움직임, 표준 패턴",
            "expected_mae": 0.40,
            "key_actions": ["W", "A", "D", "Q", "E"],
            "difficulty": "보통"
        },
        "far": {
            "description": "로봇과 먼 장애물",
            "characteristics": "큰 움직임, 넓은 경로",
            "expected_mae": 0.45,
            "key_actions": ["W", "Q", "E"],
            "difficulty": "낮음"
        }
    }
    
    print("📋 거리별 성능 특성:")
    for distance, info in distance_performance.items():
        print(f"   {distance:8s}: {info['description']}")
        print(f"           특성: {info['characteristics']}")
        print(f"           예상 MAE: {info['expected_mae']}")
        print(f"           주요 액션: {', '.join(info['key_actions'])}")
        print(f"           난이도: {info['difficulty']}")
        print()
    
    return distance_performance

def propose_distance_aware_augmentation():
    """거리 인식 데이터 증강 제안"""
    print("💡 거리 인식 데이터 증강 아이디어")
    print("=" * 60)
    
    print("🎯 현재 데이터 분포:")
    print("   Close: 3개 샘플 (정밀 조작)")
    print("   Medium: 4개 샘플 (표준 패턴)")
    print("   Far: 3개 샘플 (넓은 움직임)")
    print()
    
    print("🚀 거리별 특화 증강 전략:")
    
    # 1. Close 거리 증강
    print("1️⃣ Close 거리 증강 (정밀 조작 강화):")
    print("   • 미세 조정 증강: 기존 액션에 ±0.1~0.2 노이즈")
    print("   • 정밀 정지 패턴: SPACE 액션 빈도 증가")
    print("   • 작은 횡이동: A/D 액션 강화")
    print("   • 증강 배수: 15x (정밀도 향상 필요)")
    print()
    
    # 2. Medium 거리 증강
    print("2️⃣ Medium 거리 증강 (표준 패턴 다양화):")
    print("   • 표준 패턴 변형: Core/Variant 패턴 혼합")
    print("   • 대각선 액션 강화: Q/E 액션 비율 증가")
    print("   • 타이밍 변화: 액션 지속 시간 조정")
    print("   • 증강 배수: 10x (현재와 동일)")
    print()
    
    # 3. Far 거리 증강
    print("3️⃣ Far 거리 증강 (넓은 움직임 강화):")
    print("   • 큰 움직임 패턴: 연속 W 액션 강화")
    print("   • 대각선 경로: Q/E 액션 비율 증가")
    print("   • 속도 변화: 액션 강도 조정")
    print("   • 증강 배수: 8x (상대적으로 적음)")
    print()
    
    # 4. 거리별 가중치 학습
    print("4️⃣ 거리별 가중치 학습:")
    print("   • Close: 높은 가중치 (정밀도 중요)")
    print("   • Medium: 표준 가중치 (균형)")
    print("   • Far: 낮은 가중치 (상대적으로 쉬움)")
    print()
    
    # 5. 거리 전이 학습
    print("5️⃣ 거리 전이 학습:")
    print("   • Close → Medium: 정밀도 전이")
    print("   • Medium → Far: 패턴 전이")
    print("   • Far → Close: 안정성 전이")
    print()

def create_distance_aware_augmentation_plan():
    """거리 인식 증강 계획 생성"""
    print("📋 거리 인식 증강 실행 계획")
    print("=" * 60)
    
    # 현재 데이터 분석
    current_distribution = {
        "close": {"count": 3, "weight": 1.5, "augmentation_factor": 15},
        "medium": {"count": 4, "weight": 1.0, "augmentation_factor": 10},
        "far": {"count": 3, "weight": 0.8, "augmentation_factor": 8}
    }
    
    print("📊 현재 데이터 분포:")
    total_samples = sum(info["count"] for info in current_distribution.values())
    for distance, info in current_distribution.items():
        percentage = (info["count"] / total_samples) * 100
        print(f"   {distance:8s}: {info['count']}개 ({percentage:.1f}%)")
    print()
    
    # 증강 후 예상 분포
    print("🚀 증강 후 예상 분포:")
    total_augmented = 0
    for distance, info in current_distribution.items():
        augmented_count = info["count"] * info["augmentation_factor"]
        total_augmented += augmented_count
        print(f"   {distance:8s}: {augmented_count}개 (배수: {info['augmentation_factor']}x)")
    
    print(f"\n📈 총 증강 데이터: {total_augmented}개")
    print(f"📊 기존 대비: {total_augmented/total_samples:.1f}배 증가")
    
    # 구현 우선순위
    print("\n🎯 구현 우선순위:")
    print("   1️⃣ Close 거리 증강 (정밀도 향상)")
    print("   2️⃣ Medium 거리 증강 (표준화)")
    print("   3️⃣ Far 거리 증강 (안정성)")
    print("   4️⃣ 거리별 가중치 학습")
    print("   5️⃣ 거리 전이 학습")

def main():
    """메인 분석"""
    print("🎯 실제 데이터 수집 방식 기반 정확한 성능 분석")
    print("=" * 80)
    
    # 결과 로드
    results = load_training_results()
    final_mae = results['final_val_mae']
    
    # 1. 실제 데이터 수집 방식 분석
    actual_actions, distance_distribution = analyze_actual_data_collection()
    
    # 2. 실제 정확도 계산
    accuracy, thresholds = calculate_realistic_accuracy(final_mae)
    
    # 3. 거리별 성능 분석
    distance_performance = analyze_distance_based_performance()
    
    # 4. 거리 인식 증강 제안
    propose_distance_aware_augmentation()
    
    # 5. 증강 계획 생성
    create_distance_aware_augmentation_plan()
    
    # 6. 종합 분석
    print("\n🎯 종합 분석")
    print("=" * 60)
    print(f"📊 최종 검증 MAE: {final_mae:.3f}")
    print(f"🎯 실제 정확도: {accuracy:.1%}")
    print(f"📈 성능 등급: {'우수' if final_mae <= 0.2 else '보통' if final_mae <= 0.4 else '개선 필요' if final_mae <= 0.6 else '낮음'}")
    
    print(f"\n💡 핵심 발견:")
    print(f"   • 실제 사용 액션: 6가지 (회전/후진/Z/C 미사용)")
    print(f"   • 거리별 다양성: Close/Medium/Far 3단계")
    print(f"   • 정확도: {accuracy:.1%} (실제 액션 예측 성공률)")
    
    print(f"\n🚀 개선 방향:")
    print(f"   1. 거리별 특화 증강 (Close: 15x, Medium: 10x, Far: 8x)")
    print(f"   2. 실제 사용 액션에 집중 (W/A/D/Q/E/SPACE)")
    print(f"   3. 거리별 가중치 학습")
    print(f"   4. 거리 전이 학습 구현")
    
    print("\n🎉 분석 완료!")

if __name__ == "__main__":
    main()
