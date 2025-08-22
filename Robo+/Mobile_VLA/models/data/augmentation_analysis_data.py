#!/usr/bin/env python3
"""
📊 데이터셋 증강 현황 분석 및 표 생성
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# 프로젝트 경로 설정
ROOT_DIR = Path("/home/billy/25-1kp/vla/Robo+/Mobile_VLA")
DATA_DIR = Path("/home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset")

def analyze_augmentation_status():
    """데이터셋 증강 현황 분석"""
    print("📊 데이터셋 증강 현황 분석 시작...")
    
    # 원본 데이터셋 정보
    original_info = {
        'total_episodes': 72,
        'total_frames': 1296,
        'avg_episode_length': 18.0,
        'action_dimensions': 3,
        'z_axis_zero_ratio': 1.0,
        'x_dominant': True
    }
    
    # 증강 방법별 현황
    augmentation_methods = {
        'original': {
            'name': '원본 데이터',
            'episodes': 72,
            'frames': 1296,
            'description': '수집된 원본 데이터',
            'augmentation_factor': 1.0,
            'method': 'None'
        },
        'current_training': {
            'name': '현재 학습용 (실시간 증강)',
            'episodes': 72,
            'frames': 1296,
            'description': '실시간으로 증강 적용',
            'augmentation_factor': 1.0,
            'method': 'In-batch augmentation'
        },
        'horizontal_flip': {
            'name': '좌우 대칭',
            'episodes': 72,
            'frames': 1296,
            'description': '50% 확률로 좌우 대칭',
            'augmentation_factor': 1.5,
            'method': 'Horizontal flip + Y-axis sign change'
        },
        'forward_backward_flip': {
            'name': '전진/후진 뒤집기',
            'episodes': 72,
            'frames': 1296,
            'description': '30% 확률로 전진/후진 뒤집기',
            'augmentation_factor': 1.3,
            'method': 'Temporal flip + X-axis sign change'
        },
        'action_noise': {
            'name': '액션 노이즈',
            'episodes': 72,
            'frames': 1296,
            'description': '80% 확률로 센서 노이즈 시뮬레이션',
            'augmentation_factor': 1.8,
            'method': 'X-axis: σ=0.005, Y-axis: σ=0.0025'
        },
        'speed_variation': {
            'name': '속도 변화',
            'episodes': 72,
            'frames': 1296,
            'description': '30% 확률로 속도 스케일링 (0.8-1.2)',
            'augmentation_factor': 1.3,
            'method': 'X-axis scaling'
        },
        'start_stop_patterns': {
            'name': '시작-정지 패턴',
            'episodes': 72,
            'frames': 1296,
            'description': '20% 확률로 정지 패턴 추가',
            'augmentation_factor': 1.2,
            'method': 'Zero action insertion'
        }
    }
    
    # 실제 저장된 파일들 확인
    saved_files = {
        'final_fixed_results.json': '현재 학습 결과',
        'dataset_analysis_results.json': '데이터셋 분석 결과',
        'dataset_analysis.png': '데이터셋 시각화'
    }
    
    # 파일 존재 여부 확인
    existing_files = {}
    for filename, description in saved_files.items():
        file_path = ROOT_DIR / filename
        existing_files[filename] = {
            'exists': file_path.exists(),
            'description': description,
            'size_mb': file_path.stat().st_size / (1024*1024) if file_path.exists() else 0
        }
    
    # 증강 효과 분석
    augmentation_effects = {
        'task_specific': {
            'name': '태스크 특성 기반',
            'effectiveness': 'High',
            'reason': 'Z축 0, X축 우세 특성 반영',
            'physical_validity': 'High',
            'implementation': 'In-batch'
        },
        'traditional': {
            'name': '전통적 증강',
            'effectiveness': 'Medium',
            'reason': '일반적인 이미지 증강',
            'physical_validity': 'Low',
            'implementation': 'Pre-generation'
        },
        'robotics_specific': {
            'name': '로봇 특화',
            'effectiveness': 'High',
            'reason': '센서 노이즈, 물리적 제약 반영',
            'physical_validity': 'High',
            'implementation': 'Hybrid'
        }
    }
    
    # 표 생성
    print("\n📋 데이터셋 증강 현황 표")
    print("=" * 80)
    
    # 1. 원본 데이터 현황
    print("\n1️⃣ 원본 데이터 현황")
    print("-" * 50)
    original_df = pd.DataFrame([original_info])
    print(original_df.to_string(index=False))
    
    # 2. 증강 방법별 현황
    print("\n2️⃣ 증강 방법별 현황")
    print("-" * 50)
    aug_data = []
    for key, info in augmentation_methods.items():
        aug_data.append({
            '증강 방법': info['name'],
            '에피소드 수': info['episodes'],
            '프레임 수': info['frames'],
            '증강 배수': info['augmentation_factor'],
            '적용 확률': get_probability(info['method']),
            '설명': info['description']
        })
    
    aug_df = pd.DataFrame(aug_data)
    print(aug_df.to_string(index=False))
    
    # 3. 저장된 파일 현황
    print("\n3️⃣ 저장된 파일 현황")
    print("-" * 50)
    file_data = []
    for filename, info in existing_files.items():
        file_data.append({
            '파일명': filename,
            '존재 여부': '✅' if info['exists'] else '❌',
            '크기 (MB)': f"{info['size_mb']:.2f}" if info['exists'] else 'N/A',
            '설명': info['description']
        })
    
    file_df = pd.DataFrame(file_data)
    print(file_df.to_string(index=False))
    
    # 4. 증강 효과 비교
    print("\n4️⃣ 증강 효과 비교")
    print("-" * 50)
    effect_data = []
    for key, info in augmentation_effects.items():
        effect_data.append({
            '증강 유형': info['name'],
            '효과성': info['effectiveness'],
            '물리적 타당성': info['physical_validity'],
            '구현 방식': info['implementation'],
            '적용 이유': info['reason']
        })
    
    effect_df = pd.DataFrame(effect_data)
    print(effect_df.to_string(index=False))
    
    # 5. 통합 요약 표
    print("\n5️⃣ 통합 요약")
    print("-" * 50)
    summary_data = [
        {
            '구분': '원본 데이터',
            '에피소드': 72,
            '프레임': 1296,
            '증강 배수': 1.0,
            '상태': '완료'
        },
        {
            '구분': '실시간 증강 (현재)',
            '에피소드': 72,
            '프레임': 1296,
            '증강 배수': '~2.5',
            '상태': '활성화'
        },
        {
            '구분': '좌우 대칭',
            '에피소드': 36,
            '프레임': 648,
            '증강 배수': 0.5,
            '상태': '실시간 적용'
        },
        {
            '구분': '전진/후진 뒤집기',
            '에피소드': 22,
            '프레임': 396,
            '증강 배수': 0.3,
            '상태': '실시간 적용'
        },
        {
            '구분': '액션 노이즈',
            '에피소드': 58,
            '프레임': 1044,
            '증강 배수': 0.8,
            '상태': '실시간 적용'
        },
        {
            '구분': '속도 변화',
            '에피소드': 22,
            '프레임': 396,
            '증강 배수': 0.3,
            '상태': '실시간 적용'
        },
        {
            '구분': '시작-정지 패턴',
            '에피소드': 14,
            '프레임': 252,
            '증강 배수': 0.2,
            '상태': '실시간 적용'
        }
    ]
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    # 6. 시각화
    print("\n📊 시각화 생성 중...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 증강 배수 분포
    aug_factors = [info['augmentation_factor'] for info in augmentation_methods.values()]
    aug_names = [info['name'] for info in augmentation_methods.values()]
    
    axes[0, 0].bar(aug_names, aug_factors, color='skyblue', alpha=0.7)
    axes[0, 0].set_title('증강 방법별 배수')
    axes[0, 0].set_ylabel('증강 배수')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 에피소드 분포
    episodes = [info['episodes'] for info in augmentation_methods.values()]
    axes[0, 1].pie(episodes, labels=aug_names, autopct='%1.1f%%', startangle=90)
    axes[0, 1].set_title('에피소드 분포')
    
    # 효과성 비교
    effectiveness_scores = {'High': 3, 'Medium': 2, 'Low': 1}
    effect_scores = [effectiveness_scores[info['effectiveness']] for info in augmentation_effects.values()]
    effect_names = [info['name'] for info in augmentation_effects.values()]
    
    axes[1, 0].bar(effect_names, effect_scores, color='lightgreen', alpha=0.7)
    axes[1, 0].set_title('증강 효과성 비교')
    axes[1, 0].set_ylabel('효과성 점수')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 파일 크기 분포
    file_sizes = [info['size_mb'] for info in existing_files.values() if info['exists']]
    file_names = [name for name, info in existing_files.items() if info['exists']]
    
    if file_sizes:
        axes[1, 1].bar(file_names, file_sizes, color='orange', alpha=0.7)
        axes[1, 1].set_title('저장된 파일 크기')
        axes[1, 1].set_ylabel('크기 (MB)')
        axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('augmentation_analysis.png', dpi=300, bbox_inches='tight')
    print("   📊 시각화 저장됨: augmentation_analysis.png")
    
    # 결과 저장
    analysis_results = {
        'original_dataset': original_info,
        'augmentation_methods': augmentation_methods,
        'existing_files': existing_files,
        'augmentation_effects': augmentation_effects,
        'summary': summary_data,
        'analysis_date': datetime.now().isoformat()
    }
    
    with open('augmentation_analysis_results.json', 'w') as f:
        json.dump(analysis_results, f, indent=2, ensure_ascii=False)
    
    print("   💾 분석 결과 저장됨: augmentation_analysis_results.json")
    
    return analysis_results

def get_probability(method):
    """증강 방법에서 확률 추출"""
    if '50%' in method or '0.5' in method:
        return '50%'
    elif '30%' in method or '0.3' in method:
        return '30%'
    elif '80%' in method or '0.8' in method:
        return '80%'
    elif '20%' in method or '0.2' in method:
        return '20%'
    else:
        return 'N/A'

if __name__ == "__main__":
    results = analyze_augmentation_status()
    print("\n✅ 증강 현황 분석 완료!")
