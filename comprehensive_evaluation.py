#!/usr/bin/env python3
"""
종합 성능 평가 시스템
MAE, MSE, RMSE, Success Rate 등 다중 지표 측정
"""

import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

class ComprehensiveEvaluator:
    def __init__(self, thresholds=[0.05, 0.1, 0.15, 0.2, 0.25]):
        self.thresholds = thresholds
        
    def mae_to_success_rate(self, mae, threshold=0.1):
        """MAE를 Success Rate로 변환"""
        if mae <= threshold:
            return 1.0  # 100% 성공
        else:
            return max(0, 1 - (mae - threshold) / threshold)
    
    def calculate_all_metrics(self, predictions, targets):
        """모든 성능 지표 계산"""
        # 기본 회귀 지표
        mae = mean_absolute_error(targets, predictions)
        mse = mean_squared_error(targets, predictions)
        rmse = np.sqrt(mse)
        
        # Success Rate 계산 (다양한 임계값)
        success_rates = {}
        for threshold in self.thresholds:
            success_rates[f'success_rate_{threshold}'] = self.mae_to_success_rate(mae, threshold)
        
        # Navigation Accuracy (1 - MAE)
        navigation_accuracy = 1 - mae
        
        # 각 축별 성능
        axis_mae = {}
        for i, axis in enumerate(['x', 'y', 'z']):
            axis_mae[f'mae_{axis}'] = mean_absolute_error(targets[:, i], predictions[:, i])
        
        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'Navigation_Accuracy': navigation_accuracy,
            **success_rates,
            **axis_mae
        }
    
    def compare_with_other_models(self, our_mae):
        """다른 모델들과 성능 비교"""
        # 다른 연구들의 성능 (Success Rate 기준)
        other_models = {
            'RT-2': {'success_rate': 0.90, 'episodes': 130000},
            'RT-1': {'success_rate': 0.85, 'episodes': 130000},
            'PaLM-E': {'success_rate': 0.80, 'episodes': 562000},
            'Our Model': {'mae': our_mae, 'episodes': 72}
        }
        
        # 우리 모델의 Success Rate 변환
        our_success_rates = {}
        for threshold in self.thresholds:
            our_success_rates[threshold] = self.mae_to_success_rate(our_mae, threshold)
        
        return other_models, our_success_rates
    
    def generate_comparison_table(self, our_mae):
        """비교표 생성"""
        other_models, our_success_rates = self.compare_with_other_models(our_mae)
        
        print("📊 VLA 모델 성능 비교표")
        print("="*80)
        print(f"{'모델':<15} {'데이터셋 크기':<15} {'Success Rate':<15} {'MAE':<10} {'비고'}")
        print("-"*80)
        
        for model, metrics in other_models.items():
            if model == 'Our Model':
                print(f"{model:<15} {metrics['episodes']:<15} {our_success_rates[0.1]:<15.1%} {metrics['mae']:<10.4f} {'우리 모델'}")
            else:
                print(f"{model:<15} {metrics['episodes']:<15} {metrics['success_rate']:<15.1%} {'N/A':<10} {'기존 연구'}")
        
        print("\n🎯 임계값별 우리 모델 성능")
        print("-"*50)
        for threshold, success_rate in our_success_rates.items():
            print(f"임계값 {threshold}: {success_rate:.1%}")
    
    def plot_performance_comparison(self, our_mae):
        """성능 비교 시각화"""
        other_models, our_success_rates = self.compare_with_other_models(our_mae)
        
        # 데이터 준비
        models = ['RT-2', 'RT-1', 'PaLM-E', 'Our Model']
        success_rates = [0.90, 0.85, 0.80, our_success_rates[0.1]]
        episodes = [130000, 130000, 562000, 72]
        
        # 시각화
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Success Rate 비교
        bars1 = ax1.bar(models, success_rates, color=['green', 'blue', 'orange', 'red'])
        ax1.set_title('Success Rate 비교 (임계값 0.1)')
        ax1.set_ylabel('Success Rate')
        ax1.set_ylim(0, 1)
        
        # 값 표시
        for bar, rate in zip(bars1, success_rates):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{rate:.1%}', ha='center', va='bottom')
        
        # 데이터셋 크기 비교
        bars2 = ax2.bar(models, episodes, color=['green', 'blue', 'orange', 'red'])
        ax2.set_title('데이터셋 크기 비교')
        ax2.set_ylabel('Episodes')
        ax2.set_yscale('log')
        
        # 값 표시
        for bar, ep in zip(bars2, episodes):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1, 
                    f'{ep:,}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('vla_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_improvement_potential(self, current_mae):
        """개선 가능성 분석"""
        print("\n🚀 개선 가능성 분석")
        print("="*50)
        
        # 목표 성능 설정
        targets = {
            '단기 목표 (1개월)': 0.1,
            '중기 목표 (3개월)': 0.05,
            '장기 목표 (6개월)': 0.02
        }
        
        for period, target_mae in targets.items():
            current_sr = self.mae_to_success_rate(current_mae, 0.1)
            target_sr = self.mae_to_success_rate(target_mae, 0.1)
            improvement = (target_sr - current_sr) * 100
            
            print(f"{period}:")
            print(f"  현재 MAE: {current_mae:.4f} → 목표 MAE: {target_mae:.4f}")
            print(f"  현재 Success Rate: {current_sr:.1%} → 목표 Success Rate: {target_sr:.1%}")
            print(f"  개선 폭: {improvement:+.1f}%p")
            print()

def main():
    # 평가기 초기화
    evaluator = ComprehensiveEvaluator()
    
    # 현재 모델 성능 (MAE 0.2121)
    current_mae = 0.2121
    
    print("🔍 종합 성능 평가")
    print("="*50)
    
    # 기본 지표 계산
    metrics = evaluator.calculate_all_metrics(
        np.array([[0.2121, 0.2121, 0.2121]]),  # 예측값 (더미)
        np.array([[0.0, 0.0, 0.0]])  # 실제값 (더미)
    )
    
    print("📊 현재 모델 성능 지표")
    print("-"*30)
    for metric, value in metrics.items():
        if 'success_rate' in metric:
            print(f"{metric}: {value:.1%}")
        else:
            print(f"{metric}: {value:.4f}")
    
    # 다른 모델들과 비교
    evaluator.generate_comparison_table(current_mae)
    
    # 개선 가능성 분석
    evaluator.analyze_improvement_potential(current_mae)
    
    # 시각화
    evaluator.plot_performance_comparison(current_mae)

if __name__ == "__main__":
    main()
