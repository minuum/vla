"""
🔍 종합 모델 성능 비교 분석
모든 훈련된 모델들의 성능을 종합적으로 비교하고 분석
"""

import json
import os
from pathlib import Path

def load_evaluation_results():
    """모든 평가 결과 파일들을 로드"""
    results = {}
    
    # 평가 결과 파일들
    eval_files = [
        'optimized_2d_action_evaluation_results.json',
        'realistic_evaluation_results.json',
        'no_first_frame_evaluation_results.json',
        'advanced_mobile_vla_evaluation_results.json',
        'fixed_robovlms_evaluation_results.json'
    ]
    
    for file_path in eval_files:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    results[file_path] = data
                    print(f"✅ {file_path} 로드 완료")
            except Exception as e:
                print(f"❌ {file_path} 로드 실패: {e}")
    
    return results

def extract_performance_metrics(results):
    """성능 지표 추출"""
    metrics = {}
    
    for file_path, data in results.items():
        model_name = file_path.replace('_evaluation_results.json', '').replace('_', ' ').title()
        
        if 'optimized_2d_action' in file_path:
            # 2D 액션 모델
            metrics[model_name] = {
                'model_type': '2D_Optimized',
                'mae': data.get('avg_mae', 'N/A'),
                'rmse': data.get('avg_rmse', 'N/A'),
                'accuracy_10': data.get('success_rates', {}).get('accuracy_10', 'N/A'),
                'accuracy_5': data.get('success_rates', {}).get('accuracy_5', 'N/A'),
                'accuracy_1': data.get('success_rates', {}).get('accuracy_1', 'N/A'),
                'total_samples': data.get('total_samples', 'N/A')
            }
        
        elif 'realistic_evaluation' in file_path:
            # Realistic 평가 (첫 프레임 vs 중간 프레임)
            first_frame = data.get('first_frame_results', {})
            middle_frame = data.get('middle_frame_results', {})
            
            metrics[f"{model_name} (First Frame)"] = {
                'model_type': '3D_Realistic_First',
                'mae': first_frame.get('mae', 'N/A'),
                'rmse': first_frame.get('rmse', 'N/A'),
                'accuracy_10': first_frame.get('accuracy_0.1', 'N/A'),
                'accuracy_5': first_frame.get('accuracy_0.05', 'N/A'),
                'accuracy_1': 'N/A',
                'total_samples': len(first_frame.get('predictions', []))
            }
            
            metrics[f"{model_name} (Middle Frame)"] = {
                'model_type': '3D_Realistic_Middle',
                'mae': middle_frame.get('mae', 'N/A'),
                'rmse': middle_frame.get('rmse', 'N/A'),
                'accuracy_10': middle_frame.get('accuracy_0.1', 'N/A'),
                'accuracy_5': middle_frame.get('accuracy_0.05', 'N/A'),
                'accuracy_1': 'N/A',
                'total_samples': len(middle_frame.get('predictions', []))
            }
        
        elif 'no_first_frame' in file_path:
            # 첫 프레임 제외 모델
            random_frame = data.get('random_frame_results', {})
            middle_frame = data.get('middle_frame_results', {})
            
            metrics[f"{model_name} (Random)"] = {
                'model_type': '3D_NoFirstFrame_Random',
                'mae': random_frame.get('mae', 'N/A'),
                'rmse': random_frame.get('rmse', 'N/A'),
                'accuracy_10': random_frame.get('accuracy_0.1', 'N/A'),
                'accuracy_5': random_frame.get('accuracy_0.05', 'N/A'),
                'accuracy_1': 'N/A',
                'total_samples': len(random_frame.get('predictions', []))
            }
            
            metrics[f"{model_name} (Middle)"] = {
                'model_type': '3D_NoFirstFrame_Middle',
                'mae': middle_frame.get('mae', 'N/A'),
                'rmse': middle_frame.get('rmse', 'N/A'),
                'accuracy_10': middle_frame.get('accuracy_0.1', 'N/A'),
                'accuracy_5': middle_frame.get('accuracy_0.05', 'N/A'),
                'accuracy_1': 'N/A',
                'total_samples': len(middle_frame.get('predictions', []))
            }
        
        elif 'advanced_mobile_vla' in file_path:
            # Advanced Mobile VLA
            metrics[model_name] = {
                'model_type': '3D_Advanced',
                'mae': data.get('avg_mae', 'N/A'),
                'rmse': data.get('avg_rmse', 'N/A'),
                'accuracy_10': data.get('success_rates', {}).get('accuracy_10', 'N/A'),
                'accuracy_5': data.get('success_rates', {}).get('accuracy_5', 'N/A'),
                'accuracy_1': data.get('success_rates', {}).get('accuracy_1', 'N/A'),
                'total_samples': data.get('total_samples', 'N/A')
            }
        
        elif 'fixed_robovlms' in file_path:
            # Fixed RoboVLMs
            metrics[model_name] = {
                'model_type': '3D_Fixed_RoboVLMs',
                'mae': data.get('avg_mae', 'N/A'),
                'rmse': data.get('avg_rmse', 'N/A'),
                'accuracy_10': data.get('success_rates', {}).get('accuracy_10', 'N/A'),
                'accuracy_5': data.get('success_rates', {}).get('accuracy_5', 'N/A'),
                'accuracy_1': data.get('success_rates', {}).get('accuracy_1', 'N/A'),
                'total_samples': data.get('total_samples', 'N/A')
            }
    
    return metrics

def print_comparison_table(metrics):
    """비교 테이블 출력"""
    print("\n" + "="*120)
    print("🔍 종합 모델 성능 비교 테이블")
    print("="*120)
    
    # 헤더
    print(f"{'모델명':<35} {'타입':<20} {'MAE':<8} {'RMSE':<8} {'Acc@0.1':<8} {'Acc@0.05':<8} {'Acc@0.01':<8} {'샘플수':<8}")
    print("-" * 120)
    
    # 모델들을 성능별로 정렬 (MAE 기준)
    sorted_models = []
    for name, metric in metrics.items():
        mae = metric['mae']
        if mae != 'N/A':
            sorted_models.append((name, metric, mae))
        else:
            sorted_models.append((name, metric, float('inf')))
    
    sorted_models.sort(key=lambda x: x[2])
    
    # 결과 출력
    for i, (name, metric, _) in enumerate(sorted_models):
        rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
        
        mae_str = f"{metric['mae']:.4f}" if metric['mae'] != 'N/A' else 'N/A'
        rmse_str = f"{metric['rmse']:.4f}" if metric['rmse'] != 'N/A' else 'N/A'
        acc10_str = f"{metric['accuracy_10']:.1f}%" if metric['accuracy_10'] != 'N/A' else 'N/A'
        acc5_str = f"{metric['accuracy_5']:.1f}%" if metric['accuracy_5'] != 'N/A' else 'N/A'
        acc1_str = f"{metric['accuracy_1']:.1f}%" if metric['accuracy_1'] != 'N/A' else 'N/A'
        samples_str = f"{metric['total_samples']}" if metric['total_samples'] != 'N/A' else 'N/A'
        
        print(f"{rank} {name:<32} {metric['model_type']:<20} {mae_str:<8} {rmse_str:<8} {acc10_str:<8} {acc5_str:<8} {acc1_str:<8} {samples_str:<8}")

def analyze_model_types(metrics):
    """모델 타입별 분석"""
    print("\n" + "="*80)
    print("📊 모델 타입별 성능 분석")
    print("="*80)
    
    # 모델 타입별 그룹화
    type_groups = {}
    for name, metric in metrics.items():
        model_type = metric['model_type']
        if model_type not in type_groups:
            type_groups[model_type] = []
        type_groups[model_type].append((name, metric))
    
    # 각 타입별 분석
    for model_type, models in type_groups.items():
        print(f"\n🔍 {model_type} 모델들:")
        print("-" * 60)
        
        # MAE 기준으로 정렬
        valid_models = [(name, metric) for name, metric in models if metric['mae'] != 'N/A']
        if valid_models:
            valid_models.sort(key=lambda x: x[1]['mae'])
            
            for i, (name, metric) in enumerate(valid_models):
                rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
                print(f"{rank} {name}: MAE={metric['mae']:.4f}, RMSE={metric['rmse']:.4f}, Acc@0.1={metric['accuracy_10']:.1f}%")
        else:
            print("   유효한 성능 데이터가 없습니다.")

def generate_recommendations(metrics):
    """성능 기반 권장사항 생성"""
    print("\n" + "="*80)
    print("💡 성능 기반 권장사항")
    print("="*80)
    
    # 2D 모델과 3D 모델 분리
    d2_models = [(name, metric) for name, metric in metrics.items() if '2D' in metric['model_type']]
    d3_models = [(name, metric) for name, metric in metrics.items() if '3D' in metric['model_type']]
    
    # 최고 성능 모델 찾기
    best_2d = None
    best_3d = None
    
    if d2_models:
        best_2d = min(d2_models, key=lambda x: x[1]['mae'] if x[1]['mae'] != 'N/A' else float('inf'))
    
    if d3_models:
        best_3d = min(d3_models, key=lambda x: x[1]['mae'] if x[1]['mae'] != 'N/A' else float('inf'))
    
    print("🎯 최고 성능 모델:")
    if best_2d:
        print(f"   - 2D 모델: {best_2d[0]} (MAE: {best_2d[1]['mae']:.4f})")
    if best_3d:
        print(f"   - 3D 모델: {best_3d[0]} (MAE: {best_3d[1]['mae']:.4f})")
    
    print("\n📋 권장사항:")
    
    if best_2d and best_3d:
        if best_2d[1]['mae'] < best_3d[1]['mae']:
            print("   ✅ 2D 액션 최적화 모델이 더 나은 성능을 보입니다.")
            print("   💡 실제 로봇 제어에서는 Z축 회전이 거의 사용되지 않으므로 2D 모델을 권장합니다.")
        else:
            print("   ✅ 3D 모델이 더 나은 성능을 보입니다.")
            print("   💡 모든 액션 차원이 필요한 경우 3D 모델을 사용하세요.")
    
    print("   🔧 추가 개선 방향:")
    print("      - 데이터 증강 기법 적용")
    print("      - 앙상블 모델 사용")
    print("      - 하이퍼파라미터 튜닝")
    print("      - 더 큰 데이터셋으로 훈련")

def main():
    """메인 함수"""
    print("🔍 종합 모델 성능 비교 분석 시작!")
    
    # 평가 결과 로드
    results = load_evaluation_results()
    
    if not results:
        print("❌ 평가 결과 파일을 찾을 수 없습니다.")
        return
    
    # 성능 지표 추출
    metrics = extract_performance_metrics(results)
    
    # 비교 테이블 출력
    print_comparison_table(metrics)
    
    # 모델 타입별 분석
    analyze_model_types(metrics)
    
    # 권장사항 생성
    generate_recommendations(metrics)
    
    # 결과 저장
    comparison_results = {
        'metrics': metrics,
        'summary': {
            'total_models': len(metrics),
            'model_types': list(set(m['model_type'] for m in metrics.values())),
            'best_2d_mae': min([m['mae'] for m in metrics.values() if '2D' in m['model_type'] and m['mae'] != 'N/A'], default='N/A'),
            'best_3d_mae': min([m['mae'] for m in metrics.values() if '3D' in m['model_type'] and m['mae'] != 'N/A'], default='N/A')
        }
    }
    
    with open('comprehensive_model_comparison_results.json', 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    print(f"\n💾 비교 결과 저장 완료: comprehensive_model_comparison_results.json")
    
    print("\n" + "="*80)
    print("✅ 종합 모델 성능 비교 분석 완료!")
    print("="*80)

if __name__ == "__main__":
    main()
