#!/usr/bin/env python3
"""
Enhanced Model Comparison Analysis
Vision Resampler가 포함된 향상된 모델과 이전 모델들의 종합 비교 분석
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

def load_model_results():
    """모든 모델의 결과를 로드합니다."""
    
    # 향상된 모델 결과
    enhanced_results = {
        "Enhanced 2D with Vision Resampler": {
            "model_type": "2D_Enhanced_VisionResampler",
            "mae": 0.8040846586227417,
            "rmse": 0.8860690295696259,
            "accuracy_10": 0.0,
            "accuracy_5": 0.0,
            "accuracy_1": 0.0,
            "total_samples": 15,
            "features": ["Vision Resampler", "2D Actions", "Kosmos2 Backbone"]
        }
    }
    
    # 이전 모델 결과들
    previous_results = {
        "Optimized 2D Action": {
            "model_type": "2D_Optimized",
            "mae": 0.2919308894551268,
            "rmse": 0.48537490029934965,
            "accuracy_10": 24.836601307189543,
            "accuracy_5": 10.375816993464053,
            "accuracy_1": 0.16339869281045752,
            "total_samples": 1224,
            "features": ["2D Actions", "Kosmos2 Backbone"]
        },
        "Realistic (First Frame)": {
            "model_type": "3D_Realistic_First",
            "mae": 0.0013767265481874347,
            "rmse": 0.0020406664116308093,
            "accuracy_10": 100.0,
            "accuracy_5": 100.0,
            "accuracy_1": "N/A",
            "total_samples": 15,
            "features": ["3D Actions", "First Frame Only"]
        },
        "Realistic (Middle Frame)": {
            "model_type": "3D_Realistic_Middle",
            "mae": 0.5756955817341805,
            "rmse": 0.8074159473180771,
            "accuracy_10": 48.888888888888886,
            "accuracy_5": 48.888888888888886,
            "accuracy_1": "N/A",
            "total_samples": 15,
            "features": ["3D Actions", "Middle Frame Only"]
        },
        "No First Frame (Random)": {
            "model_type": "3D_NoFirstFrame_Random",
            "mae": 0.2405332587659359,
            "rmse": 0.42851802706718445,
            "accuracy_10": 60.0,
            "accuracy_5": 46.666666666666664,
            "accuracy_1": "N/A",
            "total_samples": 15,
            "features": ["3D Actions", "Random Frame", "No First Frame"]
        },
        "No First Frame (Middle)": {
            "model_type": "3D_NoFirstFrame_Middle",
            "mae": 0.2646177187561989,
            "rmse": 0.503977045416832,
            "accuracy_10": 62.22222222222222,
            "accuracy_5": 57.77777777777777,
            "accuracy_1": "N/A",
            "total_samples": 15,
            "features": ["3D Actions", "Middle Frame", "No First Frame"]
        }
    }
    
    return {**enhanced_results, **previous_results}

def create_comparison_table(results):
    """모델 비교 테이블을 생성합니다."""
    
    # 데이터프레임 생성
    data = []
    for model_name, metrics in results.items():
        data.append({
            'Model': model_name,
            'Type': metrics['model_type'],
            'MAE': metrics['mae'],
            'RMSE': metrics['rmse'],
            'Accuracy (10%)': metrics['accuracy_10'],
            'Accuracy (5%)': metrics['accuracy_5'],
            'Accuracy (1%)': metrics['accuracy_1'],
            'Samples': metrics['total_samples'],
            'Features': ', '.join(metrics['features'])
        })
    
    df = pd.DataFrame(data)
    
    # MAE 기준으로 정렬
    df = df.sort_values('MAE')
    
    return df

def create_visualizations(results, df):
    """시각화를 생성합니다."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Enhanced Model vs Previous Models Comparison', fontsize=16, fontweight='bold')
    
    # 1. MAE 비교
    ax1 = axes[0, 0]
    models = df['Model'].tolist()
    mae_values = df['MAE'].tolist()
    
    colors = ['red' if 'Enhanced' in model else 'blue' for model in models]
    bars = ax1.bar(range(len(models)), mae_values, color=colors, alpha=0.7)
    ax1.set_title('MAE Comparison (Lower is Better)')
    ax1.set_ylabel('MAE')
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels(models, rotation=45, ha='right')
    
    # 값 표시
    for i, (bar, value) in enumerate(zip(bars, mae_values)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 2. RMSE 비교
    ax2 = axes[0, 1]
    rmse_values = df['RMSE'].tolist()
    
    bars = ax2.bar(range(len(models)), rmse_values, color=colors, alpha=0.7)
    ax2.set_title('RMSE Comparison (Lower is Better)')
    ax2.set_ylabel('RMSE')
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels(models, rotation=45, ha='right')
    
    for i, (bar, value) in enumerate(zip(bars, rmse_values)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 3. Accuracy 비교 (10% 임계값)
    ax3 = axes[1, 0]
    acc_10_values = [float(str(v).replace('%', '')) if v != 'N/A' else 0 for v in df['Accuracy (10%)'].tolist()]
    
    bars = ax3.bar(range(len(models)), acc_10_values, color=colors, alpha=0.7)
    ax3.set_title('Accuracy (10% Threshold) Comparison (Higher is Better)')
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_xticks(range(len(models)))
    ax3.set_xticklabels(models, rotation=45, ha='right')
    
    for i, (bar, value) in enumerate(zip(bars, acc_10_values)):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{value:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # 4. 모델 타입별 분포
    ax4 = axes[1, 1]
    model_types = df['Type'].value_counts()
    ax4.pie(model_types.values, labels=model_types.index, autopct='%1.1f%%', startangle=90)
    ax4.set_title('Model Types Distribution')
    
    plt.tight_layout()
    plt.savefig('enhanced_model_comparison_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_analysis_report(df, results):
    """분석 보고서를 생성합니다."""
    
    report = {
        "analysis_summary": {
            "total_models": len(results),
            "enhanced_model_rank": df[df['Model'].str.contains('Enhanced')].index[0] + 1,
            "best_mae_model": df.iloc[0]['Model'],
            "best_mae_value": df.iloc[0]['MAE'],
            "worst_mae_model": df.iloc[-1]['Model'],
            "worst_mae_value": df.iloc[-1]['MAE']
        },
        "enhanced_model_analysis": {
            "mae_rank": df[df['Model'].str.contains('Enhanced')].index[0] + 1,
            "mae_value": df[df['Model'].str.contains('Enhanced')]['MAE'].iloc[0],
            "rmse_value": df[df['Model'].str.contains('Enhanced')]['RMSE'].iloc[0],
            "accuracy_10": df[df['Model'].str.contains('Enhanced')]['Accuracy (10%)'].iloc[0],
            "unique_features": ["Vision Resampler"],
            "performance_analysis": "Vision Resampler가 포함된 향상된 모델의 성능 분석"
        },
        "key_findings": [
            "Vision Resampler 모델이 가장 높은 MAE를 보임 (0.804)",
            "2D 최적화 모델이 가장 낮은 MAE를 보임 (0.292)",
            "첫 프레임 모델이 100% 정확도를 보이지만 실제 의미는 제한적",
            "중간 프레임 모델들이 더 현실적인 성능을 보임",
            "Vision Resampler의 추가가 예상과 다른 결과를 보임"
        ],
        "recommendations": [
            "Vision Resampler 구현을 재검토하고 최적화 필요",
            "2D 액션 최적화가 가장 효과적인 접근법임을 확인",
            "더 큰 데이터셋에서 Vision Resampler 효과 재평가 필요",
            "하이퍼파라미터 튜닝으로 Vision Resampler 성능 개선 가능성",
            "다른 RoboVLMs 고급 기능들과의 조합 실험 필요"
        ]
    }
    
    return report

def main():
    """메인 분석 함수"""
    
    print("🔍 Enhanced Model Comparison Analysis")
    print("=" * 50)
    
    # 결과 로드
    results = load_model_results()
    
    # 비교 테이블 생성
    df = create_comparison_table(results)
    
    print("\n📊 Model Comparison Table:")
    print(df.to_string(index=False))
    
    # 시각화 생성
    print("\n📈 Creating visualizations...")
    create_visualizations(results, df)
    
    # 분석 보고서 생성
    report = generate_analysis_report(df, results)
    
    print("\n📋 Analysis Report:")
    print(f"Total Models: {report['analysis_summary']['total_models']}")
    print(f"Enhanced Model Rank: {report['analysis_summary']['enhanced_model_rank']}")
    print(f"Best MAE Model: {report['analysis_summary']['best_mae_model']} ({report['analysis_summary']['best_mae_value']:.3f})")
    print(f"Worst MAE Model: {report['analysis_summary']['worst_mae_model']} ({report['analysis_summary']['worst_mae_value']:.3f})")
    
    print("\n🔍 Key Findings:")
    for finding in report['key_findings']:
        print(f"  • {finding}")
    
    print("\n💡 Recommendations:")
    for rec in report['recommendations']:
        print(f"  • {rec}")
    
    # 결과 저장
    with open('enhanced_model_comparison_results.json', 'w') as f:
        json.dump({
            'comparison_table': df.to_dict('records'),
            'analysis_report': report,
            'raw_results': results
        }, f, indent=2, default=str)
    
    print(f"\n✅ Analysis completed! Results saved to:")
    print(f"  - enhanced_model_comparison_analysis.png")
    print(f"  - enhanced_model_comparison_results.json")

if __name__ == "__main__":
    main()
