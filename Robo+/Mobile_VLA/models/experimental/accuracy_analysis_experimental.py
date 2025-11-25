#!/usr/bin/env python3
"""
🎯 Mobile VLA 모델 정확도 분석
실제 예측 정확도를 자세히 분석
"""

import json
import numpy as np
import os
from pathlib import Path

def load_test_results(file_path):
    """테스트 결과 로드"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except:
        return None

def calculate_accuracy_at_threshold(predictions, targets, threshold):
    """특정 임계값에서의 정확도 계산"""
    if len(predictions) == 0:
        return 0.0
    
    correct = 0
    total = len(predictions)
    
    for pred, target in zip(predictions, targets):
        # L2 거리 계산
        distance = np.sqrt((pred[0] - target[0])**2 + (pred[1] - target[1])**2)
        if distance <= threshold:
            correct += 1
    
    return (correct / total) * 100

def analyze_accuracy_distribution(predictions, targets):
    """정확도 분포 분석"""
    distances = []
    for pred, target in zip(predictions, targets):
        distance = np.sqrt((pred[0] - target[0])**2 + (pred[1] - target[1])**2)
        distances.append(distance)
    
    distances = np.array(distances)
    
    # 다양한 임계값에서의 정확도
    thresholds = [0.1, 0.2, 0.3, 0.5, 1.0, 1.5, 2.0]
    accuracies = {}
    
    for threshold in thresholds:
        accuracies[f'threshold_{threshold}'] = calculate_accuracy_at_threshold(predictions, targets, threshold)
    
    # 통계 정보
    stats = {
        'mean_distance': float(np.mean(distances)),
        'median_distance': float(np.median(distances)),
        'std_distance': float(np.std(distances)),
        'min_distance': float(np.min(distances)),
        'max_distance': float(np.max(distances)),
        'percentile_25': float(np.percentile(distances, 25)),
        'percentile_75': float(np.percentile(distances, 75))
    }
    
    return accuracies, stats, distances

def main():
    print("=" * 80)
    print("🎯 Mobile VLA 모델 정확도 상세 분석")
    print("=" * 80)
    
    # 실제 데이터 결과 수집
    real_results = {}
    
    # 각 케이스의 결과 로드
    cases = {
        'Case 1': "models/immediate/case1_real_results/test_results.json",
        'Case 2': "models/short_term/case2_real_results/test_results.json", 
        'Case 3': "models/medium_term/case3_real_results/test_results.json",
        'Case 4': "models/long_term/case4_real_results/test_results.json",
        'Case 5': "models/future/case5_real_results/test_results.json"
    }
    
    for case_name, file_path in cases.items():
        if os.path.exists(file_path):
            result = load_test_results(file_path)
            if result:
                real_results[case_name] = result
    
    print("\n📊 각 케이스별 정확도 분석:")
    print("-" * 80)
    
    # 결과 정렬 (MAE 기준)
    sorted_results = sorted(real_results.items(), key=lambda x: x[1]['test_mae'])
    
    for i, (case_name, result) in enumerate(sorted_results):
        rank_emoji = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][i]
        print(f"\n{rank_emoji} {case_name} (MAE: {result['test_mae']:.6f}):")
        
        # 다양한 임계값에서의 정확도
        thresholds = [0.1, 0.2, 0.3, 0.5, 1.0, 1.5, 2.0]
        print("   📏 임계값별 정확도:")
        
        for threshold in thresholds:
            accuracy_key = f'accuracy_{threshold}'
            if accuracy_key in result.get('accuracies', {}):
                accuracy = result['accuracies'][accuracy_key]
                print(f"      - {threshold:.1f} 단위: {accuracy:.2f}%")
            else:
                print(f"      - {threshold:.1f} 단위: 데이터 없음")
        
        # R² 점수
        r2_x = result.get('r2_scores', {}).get('linear_x_r2', 0)
        r2_y = result.get('r2_scores', {}).get('linear_y_r2', 0)
        print(f"   📈 R² 점수: X={r2_x:.6f}, Y={r2_y:.6f}")
        
        # 상관관계
        corr_x = result.get('correlations', {}).get('linear_x_correlation', 0)
        corr_y = result.get('correlations', {}).get('linear_y_correlation', 0)
        print(f"   🔗 상관관계: X={corr_x:.6f}, Y={corr_y:.6f}")
    
    print("\n🎯 정확도 해석 가이드:")
    print("-" * 80)
    print("📏 임계값별 의미:")
    print("   - 0.1 단위: 매우 정확한 예측 (10cm 이내)")
    print("   - 0.2 단위: 정확한 예측 (20cm 이내)")
    print("   - 0.3 단위: 보통 정확한 예측 (30cm 이내)")
    print("   - 0.5 단위: 허용 가능한 예측 (50cm 이내)")
    print("   - 1.0 단위: 대략적인 예측 (1m 이내)")
    print("   - 1.5 단위: 근사한 예측 (1.5m 이내)")
    print("   - 2.0 단위: 대략적인 예측 (2m 이내)")
    
    print("\n📊 성능 등급:")
    print("   - 90%+ : 우수 (Excellent)")
    print("   - 80-90% : 양호 (Good)")
    print("   - 70-80% : 보통 (Fair)")
    print("   - 50-70% : 미흡 (Poor)")
    print("   - 0-50% : 매우 미흡 (Very Poor)")
    
    print("\n💡 현재 모델 성능 평가:")
    print("-" * 80)
    
    # 현재 0.3 임계값에서의 성능 분석
    print("🔍 0.3 단위 임계값에서의 성능:")
    for case_name, result in sorted_results:
        accuracy_03 = result.get('accuracies', {}).get('accuracy_0.3', 0)
        mae = result['test_mae']
        print(f"   - {case_name}: {accuracy_03:.2f}% (MAE: {mae:.6f})")
    
    print("\n⚠️ 문제점 분석:")
    print("   - 모든 모델이 0.3 단위에서 0% 정확도")
    print("   - 이는 예측 오차가 30cm를 초과함을 의미")
    print("   - 실제 로봇 주행에서는 매우 큰 오차")
    
    print("\n🎯 개선 방향:")
    print("   1. 더 많은 훈련 데이터 수집")
    print("   2. 하이퍼파라미터 튜닝")
    print("   3. 모델 아키텍처 개선")
    print("   4. 데이터 증강 기법 적용")
    print("   5. 앙상블 모델 고려")
    
    print("\n📈 실제 사용 가능성:")
    print("-" * 80)
    print("🔴 현재 상태: 실제 로봇 주행에 부적합")
    print("   - 30cm 오차는 로봇 주행에서 매우 위험")
    print("   - 장애물 회피나 정밀 주행 불가능")
    print("   - 추가 개발 및 개선 필요")
    
    print("\n🟡 개선 후 기대:")
    print("   - 0.5 단위에서 70%+ 정확도 달성 시")
    print("   - 기본적인 주행 태스크 가능")
    print("   - 0.3 단위에서 50%+ 정확도 달성 시")
    print("   - 정밀 주행 및 장애물 회피 가능")
    
    print("\n" + "=" * 80)
    print("✅ 정확도 분석 완료!")
    print("=" * 80)

if __name__ == "__main__":
    main()
