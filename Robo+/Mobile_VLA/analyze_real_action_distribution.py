"""
🔍 실제 액션 분포 분석
데이터 수집기에서 정의된 WASD 액션과 실제 수집된 데이터의 액션 분포를 분석
"""

import h5py
import numpy as np
from pathlib import Path
import json

def analyze_action_distribution():
    """실제 수집된 데이터의 액션 분포 분석"""
    
    print("🔍 실제 액션 분포 분석")
    print("=" * 60)
    
    # 데이터 수집기에서 정의된 WASD 액션 매핑
    WASD_TO_CONTINUOUS = {
        'w': {"linear_x": 1.15, "linear_y": 0.0, "angular_z": 0.0},
        'a': {"linear_x": 0.0, "linear_y": 1.15, "angular_z": 0.0},
        's': {"linear_x": -1.15, "linear_y": 0.0, "angular_z": 0.0},
        'd': {"linear_x": 0.0, "linear_y": -1.15, "angular_z": 0.0},
        'q': {"linear_x": 1.15, "linear_y": 1.15, "angular_z": 0.0},
        'e': {"linear_x": 1.15, "linear_y": -1.15, "angular_z": 0.0},
        'z': {"linear_x": -1.15, "linear_y": 1.15, "angular_z": 0.0},
        'c': {"linear_x": -1.15, "linear_y": -1.15, "angular_z": 0.0},
        'r': {"linear_x": 0.0, "linear_y": 0.0, "angular_z": 1.15},
        't': {"linear_x": 0.0, "linear_y": 0.0, "angular_z": -1.15},
        ' ': {"linear_x": 0.0, "linear_y": 0.0, "angular_z": 0.0}
    }
    
    print("📋 데이터 수집기 WASD 액션 정의:")
    print("   🚶 이동 액션: W(전진), A(좌측), S(후진), D(우측)")
    print("   🚶‍♂️ 대각선 액션: Q(전진+좌측), E(전진+우측), Z(후진+좌측), C(후진+우측)")
    print("   🔄 회전 액션: R(좌회전), T(우회전)")
    print("   🛑 정지 액션: SPACE(정지)")
    print()
    
    # 실제 데이터 분석
    data_path = Path("../../ROS_action/mobile_vla_dataset")
    
    if not data_path.exists():
        print(f"❌ 데이터 경로가 존재하지 않습니다: {data_path}")
        return
    
    h5_files = list(data_path.glob("*.h5"))
    print(f"📁 발견된 H5 파일: {len(h5_files)}개")
    
    if len(h5_files) == 0:
        print("❌ 분석할 H5 파일이 없습니다.")
        return
    
    # 액션 통계 초기화
    action_counts = {}
    unique_actions = set()
    total_frames = 0
    z_axis_usage = 0
    
    # 각 H5 파일 분석
    for h5_file in h5_files[:10]:  # 처음 10개 파일만 분석
        try:
            with h5py.File(h5_file, 'r') as f:
                if 'actions' in f:
                    actions = f['actions'][:]  # [num_frames, 3]
                    total_frames += len(actions)
                    
                    for action in actions:
                        # 액션을 문자열로 변환하여 카운트
                        action_str = f"({action[0]:.2f}, {action[1]:.2f}, {action[2]:.2f})"
                        action_counts[action_str] = action_counts.get(action_str, 0) + 1
                        unique_actions.add(action_str)
                        
                        # Z축 사용 여부 확인
                        if abs(action[2]) > 0.01:  # angular_z가 0이 아닌 경우
                            z_axis_usage += 1
                            
        except Exception as e:
            print(f"⚠️ {h5_file.name} 분석 실패: {e}")
    
    print(f"\n📊 액션 분포 분석 결과:")
    print(f"   - 총 프레임 수: {total_frames}")
    print(f"   - 고유 액션 수: {len(unique_actions)}")
    print(f"   - Z축 사용 프레임: {z_axis_usage} ({z_axis_usage/total_frames*100:.1f}%)")
    
    # 가장 많이 사용된 액션들
    print(f"\n🏆 가장 많이 사용된 액션 (상위 10개):")
    sorted_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)
    
    for i, (action_str, count) in enumerate(sorted_actions[:10]):
        percentage = count / total_frames * 100
        print(f"   {i+1:2d}. {action_str}: {count:4d}회 ({percentage:5.1f}%)")
    
    # WASD 액션과 매칭되는 액션들 찾기
    print(f"\n🎯 WASD 액션과 매칭되는 실제 액션들:")
    wasd_matches = {}
    
    for key, wasd_action in WASD_TO_CONTINUOUS.items():
        wasd_str = f"({wasd_action['linear_x']:.2f}, {wasd_action['linear_y']:.2f}, {wasd_action['angular_z']:.2f})"
        count = action_counts.get(wasd_str, 0)
        percentage = count / total_frames * 100 if total_frames > 0 else 0
        wasd_matches[key] = {
            'action': wasd_action,
            'count': count,
            'percentage': percentage
        }
        print(f"   {key.upper():>2}: {wasd_str} → {count:4d}회 ({percentage:5.1f}%)")
    
    # Z축 사용 분석
    print(f"\n🔄 Z축 (회전) 사용 분석:")
    z_actions = [k for k, v in action_counts.items() if abs(float(k.split(',')[2].strip(')')) > 0.01)]
    z_total = sum(action_counts[k] for k in z_actions)
    print(f"   - Z축 사용 액션 수: {len(z_actions)}개")
    print(f"   - Z축 사용 총 프레임: {z_total}회 ({z_total/total_frames*100:.1f}%)")
    
    if z_actions:
        print(f"   - Z축 사용 액션들:")
        for action_str in z_actions[:5]:  # 상위 5개만
            count = action_counts[action_str]
            print(f"     • {action_str}: {count}회")
    
    # 결과 저장
    analysis_result = {
        'total_frames': total_frames,
        'unique_actions': len(unique_actions),
        'z_axis_usage': {
            'frames': z_axis_usage,
            'percentage': z_axis_usage/total_frames*100 if total_frames > 0 else 0
        },
        'top_actions': sorted_actions[:10],
        'wasd_matches': wasd_matches,
        'z_actions': z_actions[:10] if z_actions else []
    }
    
    with open('action_distribution_analysis.json', 'w') as f:
        json.dump(analysis_result, f, indent=2)
    
    print(f"\n💾 분석 결과 저장: action_distribution_analysis.json")
    
    return analysis_result

def recommend_model_improvements(analysis_result):
    """분석 결과를 바탕으로 모델 개선 제안"""
    
    print(f"\n🎯 모델 개선 제안:")
    print("=" * 60)
    
    z_usage_percentage = analysis_result['z_axis_usage']['percentage']
    
    if z_usage_percentage < 5:
        print(f"✅ Z축 사용률이 낮음 ({z_usage_percentage:.1f}%)")
        print("   💡 제안사항:")
        print("   • Z축 가중치를 더 낮게 설정 (현재 0.05 → 0.01)")
        print("   • Z축 손실 함수에서 제외 고려")
        print("   • 2D 액션 모델로 단순화 고려")
    elif z_usage_percentage < 20:
        print(f"⚠️ Z축 사용률이 중간 ({z_usage_percentage:.1f}%)")
        print("   💡 제안사항:")
        print("   • Z축 가중치를 적당히 조정 (현재 0.05 → 0.02)")
        print("   • Z축 예측 정확도 별도 모니터링")
    else:
        print(f"❌ Z축 사용률이 높음 ({z_usage_percentage:.1f}%)")
        print("   💡 제안사항:")
        print("   • Z축 가중치를 높게 설정 (현재 0.05 → 0.1)")
        print("   • 회전 액션 예측 성능 개선 필요")
    
    # 가장 많이 사용된 액션 분석
    top_action = analysis_result['top_actions'][0] if analysis_result['top_actions'] else None
    if top_action:
        action_str, count = top_action
        percentage = count / analysis_result['total_frames'] * 100
        print(f"\n📊 가장 많이 사용된 액션: {action_str} ({percentage:.1f}%)")
        
        if percentage > 50:
            print("   ⚠️ 특정 액션에 과도하게 편중됨")
            print("   💡 제안사항:")
            print("   • 데이터 수집 다양성 개선 필요")
            print("   • 액션 밸런싱을 위한 추가 수집 고려")
    
    print(f"\n🔧 구체적 개선 방안:")
    print("1. Z축 가중치 조정:")
    print("   - 현재: z_axis_weight = 0.05")
    print("   - 제안: z_axis_weight = 0.01 (Z축 사용률이 낮은 경우)")
    
    print("2. 손실 함수 개선:")
    print("   - Z축 사용률이 낮으면 Z축 손실을 별도로 계산")
    print("   - X, Y축에 집중한 학습")
    
    print("3. 모델 구조 최적화:")
    print("   - 2D 액션 전용 모델 고려")
    print("   - Z축 예측을 선택적으로 활성화")

if __name__ == "__main__":
    analysis_result = analyze_action_distribution()
    if analysis_result:
        recommend_model_improvements(analysis_result)
