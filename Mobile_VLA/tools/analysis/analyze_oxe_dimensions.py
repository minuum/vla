import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # GUI 없이 이미지만 저장
import matplotlib.pyplot as plt

def analyze_oxe_dimensions():
    """OXE Dataset의 액션과 상태 차원 분석"""
    print("=== OXE Dataset 차원 분석 ===")
    
    # 데이터 로드
    data = np.load('cup.npy', allow_pickle=True)
    sample = data[0]
    
    print(f"태스크: {sample['task'][0][0]}")
    print("\n=== 액션 분석 (8차원) ===")
    actions = sample['action']
    print(f"액션 형태: {actions.shape}")
    
    # 액션의 각 차원별 통계
    action_stats = pd.DataFrame({
        'min': actions.min(axis=0),
        'max': actions.max(axis=0), 
        'mean': actions.mean(axis=0),
        'std': actions.std(axis=0)
    }, index=[f'action_{i}' for i in range(8)])
    
    print("\n액션 차원별 통계:")
    print(action_stats.round(4))
    
    # 액션 변화량 분석
    action_deltas = np.diff(actions, axis=0)
    print(f"\n액션 변화량 통계 (차분):")
    action_delta_stats = pd.DataFrame({
        'min': action_deltas.min(axis=0),
        'max': action_deltas.max(axis=0),
        'mean': action_deltas.mean(axis=0),
        'std': action_deltas.std(axis=0)
    }, index=[f'action_{i}' for i in range(8)])
    print(action_delta_stats.round(4))
    
    print("\n=== 로봇 상태 분석 (15차원) ===")
    robot_states = sample['robot_state']
    print(f"로봇 상태 형태: {robot_states.shape}")
    
    # 로봇 상태의 각 차원별 통계
    state_stats = pd.DataFrame({
        'min': robot_states.min(axis=0),
        'max': robot_states.max(axis=0),
        'mean': robot_states.mean(axis=0),
        'std': robot_states.std(axis=0)
    }, index=[f'state_{i}' for i in range(15)])
    
    print("\n로봇 상태 차원별 통계:")
    print(state_stats.round(4))
    
    # 상태 변화량 분석
    state_deltas = np.diff(robot_states, axis=0)
    print(f"\n로봇 상태 변화량 통계:")
    state_delta_stats = pd.DataFrame({
        'min': state_deltas.min(axis=0),
        'max': state_deltas.max(axis=0),
        'mean': state_deltas.mean(axis=0),
        'std': state_deltas.std(axis=0)
    }, index=[f'state_{i}' for i in range(15)])
    print(state_delta_stats.round(4))
    
    # 추가 분석: 특정 차원들의 패턴 확인
    print("\n=== 패턴 분석 ===")
    
    # 0값이 많은 차원들 확인 (그리퍼 상태일 가능성)
    zero_ratio_actions = (actions == 0).mean(axis=0)
    zero_ratio_states = (robot_states == 0).mean(axis=0)
    
    print("액션에서 0값 비율이 높은 차원들:")
    for i, ratio in enumerate(zero_ratio_actions):
        if ratio > 0.5:
            print(f"  action_{i}: {ratio:.2%} 가 0값")
    
    print("\n로봇 상태에서 0값 비율이 높은 차원들:")
    for i, ratio in enumerate(zero_ratio_states):
        if ratio > 0.5:
            print(f"  state_{i}: {ratio:.2%} 가 0값")
    
    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 액션 시계열
    axes[0, 0].plot(actions[:, :6])  # 처음 6개 액션 차원
    axes[0, 0].set_title('Actions (첫 6차원) over Time')
    axes[0, 0].set_xlabel('Timestep')
    axes[0, 0].legend([f'action_{i}' for i in range(6)], bbox_to_anchor=(1.05, 1))
    
    # 그리퍼 액션 (마지막 2차원)
    axes[0, 1].plot(actions[:, 6:])
    axes[0, 1].set_title('Gripper Actions (마지막 2차원)')
    axes[0, 1].set_xlabel('Timestep')
    axes[0, 1].legend(['action_6', 'action_7'])
    
    # 로봇 상태 (처음 7개 - 관절 각도로 추정)
    axes[1, 0].plot(robot_states[:, :7])
    axes[1, 0].set_title('Robot Joint States (첫 7차원)')
    axes[1, 0].set_xlabel('Timestep')
    axes[1, 0].legend([f'state_{i}' for i in range(7)], bbox_to_anchor=(1.05, 1))
    
    # 그리퍼 상태 및 기타
    axes[1, 1].plot(robot_states[:, 7:])
    axes[1, 1].set_title('Gripper & Other States (나머지 8차원)')
    axes[1, 1].set_xlabel('Timestep') 
    axes[1, 1].legend([f'state_{i}' for i in range(7, 15)], bbox_to_anchor=(1.05, 1))
    
    plt.tight_layout()
    plt.savefig('oxe_dimensions_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 상세 통계를 CSV로 저장
    action_stats.to_csv('action_dimensions_stats.csv')
    state_stats.to_csv('robot_state_dimensions_stats.csv')
    
    print(f"\n시각화 저장: oxe_dimensions_analysis.png")
    print(f"통계 저장: action_dimensions_stats.csv, robot_state_dimensions_stats.csv")

if __name__ == "__main__":
    analyze_oxe_dimensions()