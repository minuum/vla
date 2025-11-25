#!/usr/bin/env python3
"""
H5 에피소드 파일 분석 및 통계 생성
- Trajectory 분석 (18프레임의 wasd 액션)
- Task별 일관성 확인
- 시각화
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import glob
import json
from datetime import datetime

def extract_task_name(filename):
    """파일명에서 task 이름 추출 (시간대 및 타임스탬프 제외)"""
    # episode_20251114_142640_1box_hori_left_core_medium.h5
    # -> 1box_hori_left_core_medium
    # episode_20251114_145248_1box_hori_left_core_medium_evening.h5
    # -> 1box_hori_left_core_medium (evening 제거)
    basename = Path(filename).stem
    parts = basename.split('_')
    
    # episode_날짜_시간_task... 형식
    # 처음 3개 제거 (episode, 날짜, 시간)
    if len(parts) >= 4 and parts[0] == 'episode':
        task_parts = parts[3:]
    # 날짜_시간_task... 형식 (이미 episode 제거됨)
    elif len(parts) >= 3:
        # 날짜와 시간 부분 제거 (처음 2개)
        task_parts = parts[2:]
    else:
        return 'unknown'
    
    # 시간대 제거 (evening, morning 등)
    if task_parts and task_parts[-1] in ['evening', 'morning', 'afternoon', 'night', 'dawn']:
        task_parts = task_parts[:-1]
    
    return '_'.join(task_parts) if task_parts else 'unknown'

def analyze_episode(h5_path):
    """단일 에피소드 분석"""
    with h5py.File(h5_path, 'r') as f:
        # 데이터 확인
        if 'actions' not in f:
            print(f"⚠️  {h5_path}: 'actions' 키 없음")
            return None
        
        actions = f['actions'][:]  # (N, 2) 또는 (N, 7)
        images = f['images'][:] if 'images' in f else None
        
        # 액션 차원 확인
        if len(actions.shape) == 1:
            actions = actions.reshape(-1, 1)
        
        # 2D 액션만 추출 (linear_x, linear_y)
        if actions.shape[1] >= 2:
            actions_2d = actions[:, :2]
        else:
            actions_2d = actions
        
        return {
            'filename': Path(h5_path).name,
            'num_frames': len(actions),
            'actions': actions_2d,
            'num_images': len(images) if images is not None else 0,
            'task_name': extract_task_name(h5_path)
        }

def analyze_trajectory(actions, window_size=18):
    """Trajectory 분석 - 18프레임 단위로 액션 패턴 확인"""
    if len(actions) < window_size:
        return None
    
    # 18프레임 단위로 나누기
    num_windows = len(actions) // window_size
    trajectories = []
    
    for i in range(num_windows):
        window_actions = actions[i*window_size:(i+1)*window_size]
        trajectories.append(window_actions)
    
    return trajectories

def compare_trajectories(traj1, traj2, threshold=0.01):
    """두 trajectory가 같은지 비교 (threshold 내에서)"""
    if traj1.shape != traj2.shape:
        return False
    
    diff = np.abs(traj1 - traj2)
    max_diff = np.max(diff)
    return max_diff < threshold

def categorize_action(action):
    """액션을 wasd 카테고리로 분류 (데이터 수집 코드 기준)
    
    데이터 수집 코드 매핑 (mobile_vla_data_collector.py):
    - 'w': linear_x=1.15, linear_y=0.0 (전진)
    - 'a': linear_x=0.0, linear_y=1.15 (좌)
    - 'd': linear_x=0.0, linear_y=-1.15 (우)
    - 's': linear_x=-1.15, linear_y=0.0 (후진)
    - 'q': linear_x=1.15, linear_y=1.15 (전진+좌)
    - 'e': linear_x=1.15, linear_y=-1.15 (전진+우)
    - 'z': linear_x=-1.15, linear_y=1.15 (후진+좌)
    - 'c': linear_x=-1.15, linear_y=-1.15 (후진+우)
    - ' ': linear_x=0.0, linear_y=0.0 (정지)
    """
    linear_x, linear_y = action[0], action[1]
    
    # 임계값 설정
    thresh = 0.1
    
    # 정지
    if abs(linear_x) < thresh and abs(linear_y) < thresh:
        return 'S'  # Stop (스페이스바)
    # 대각선 액션 우선 처리
    elif linear_x > thresh and linear_y > thresh:
        return 'Q'  # 전진+좌 (q 키)
    elif linear_x > thresh and linear_y < -thresh:
        return 'E'  # 전진+우 (e 키)
    elif linear_x < -thresh and linear_y > thresh:
        return 'Z'  # 후진+좌 (z 키)
    elif linear_x < -thresh and linear_y < -thresh:
        return 'C'  # 후진+우 (c 키)
    # 단일 방향 액션
    elif linear_x > thresh and abs(linear_y) < thresh:
        return 'W'  # Forward (w 키)
    elif linear_x < -thresh and abs(linear_y) < thresh:
        return 'S'  # Backward (s 키, 정지로 처리)
    elif abs(linear_x) < thresh and linear_y > thresh:
        return 'A'  # Left (a 키: linear_y=1.15)
    elif abs(linear_x) < thresh and linear_y < -thresh:
        return 'D'  # Right (d 키: linear_y=-1.15)
    else:
        return '?'

def trajectory_to_string(trajectory):
    """Trajectory를 문자열로 변환 (WASD)"""
    return ''.join([categorize_action(a) for a in trajectory])

def main():
    data_dir = Path("ROS_action/mobile_vla_dataset")
    
    print("=" * 60)
    print("H5 에피소드 파일 분석 시작")
    print("=" * 60)
    print()
    
    # 모든 H5 파일 찾기
    h5_files = sorted(glob.glob(str(data_dir / "episode_*.h5")))
    print(f"📁 총 {len(h5_files)}개 에피소드 파일 발견")
    print()
    
    # 에피소드 분석
    episodes = []
    task_groups = defaultdict(list)
    
    for h5_file in h5_files:
        result = analyze_episode(h5_file)
        if result is None:
            continue
        
        episodes.append(result)
        task_groups[result['task_name']].append(result)
    
    print(f"✅ {len(episodes)}개 에피소드 분석 완료")
    print()
    
    # Task별 통계
    print("=" * 60)
    print("Task별 통계")
    print("=" * 60)
    
    task_stats = {}
    for task_name, task_episodes in task_groups.items():
        print(f"\n📋 Task: {task_name}")
        print(f"   에피소드 수: {len(task_episodes)}")
        
        frame_counts = [ep['num_frames'] for ep in task_episodes]
        print(f"   프레임 수: 평균 {np.mean(frame_counts):.1f}, 최소 {min(frame_counts)}, 최대 {max(frame_counts)}")
        
        # Trajectory 분석
        all_trajectories = []
        for ep in task_episodes:
            trajs = analyze_trajectory(ep['actions'], window_size=18)
            if trajs:
                all_trajectories.extend(trajs)
        
        if all_trajectories:
            # Trajectory 문자열 변환
            traj_strings = [trajectory_to_string(traj) for traj in all_trajectories]
            unique_trajs = set(traj_strings)
            
            print(f"   총 Trajectory 수: {len(traj_strings)}")
            print(f"   고유 Trajectory 수: {len(unique_trajs)}")
            
            # 가장 흔한 trajectory
            from collections import Counter
            traj_counts = Counter(traj_strings)
            most_common = traj_counts.most_common(5)
            print(f"   가장 흔한 Trajectory:")
            for traj, count in most_common:
                print(f"     '{traj}': {count}회 ({count/len(traj_strings)*100:.1f}%)")
            
            # 일관성 확인
            consistency = len(unique_trajs) / len(traj_strings) if traj_strings else 0
            print(f"   일관성 점수: {1-consistency:.2%} (낮을수록 일관적)")
            
            task_stats[task_name] = {
                'num_episodes': len(task_episodes),
                'num_trajectories': len(traj_strings),
                'num_unique_trajectories': len(unique_trajs),
                'consistency': 1 - consistency,
                'most_common': most_common[:3]
            }
        else:
            print(f"   ⚠️  Trajectory 분석 불가 (프레임 수 부족)")
    
    # 전체 통계
    print("\n" + "=" * 60)
    print("전체 통계")
    print("=" * 60)
    
    all_frame_counts = [ep['num_frames'] for ep in episodes]
    print(f"\n총 에피소드: {len(episodes)}")
    print(f"총 프레임: {sum(all_frame_counts)}")
    print(f"평균 프레임/에피소드: {np.mean(all_frame_counts):.1f}")
    print(f"Task 종류: {len(task_groups)}")
    
    # 시각화
    print("\n" + "=" * 60)
    print("시각화 생성 중...")
    print("=" * 60)
    
    # 1. Task별 에피소드 수
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Task별 에피소드 수
    ax = axes[0, 0]
    task_names = list(task_groups.keys())
    episode_counts = [len(task_groups[t]) for t in task_names]
    ax.barh(task_names, episode_counts)
    ax.set_xlabel('에피소드 수')
    ax.set_title('Task별 에피소드 수')
    ax.grid(axis='x', alpha=0.3)
    
    # Task별 프레임 수 분포
    ax = axes[0, 1]
    for task_name in task_names[:10]:  # 상위 10개만
        frame_counts = [ep['num_frames'] for ep in task_groups[task_name]]
        ax.hist(frame_counts, alpha=0.5, label=task_name, bins=20)
    ax.set_xlabel('프레임 수')
    ax.set_ylabel('빈도')
    ax.set_title('Task별 프레임 수 분포')
    ax.legend(fontsize=8)
    
    # Task별 일관성 점수
    ax = axes[1, 0]
    if task_stats:
        tasks = list(task_stats.keys())
        consistencies = [task_stats[t]['consistency'] for t in tasks]
        ax.barh(tasks, consistencies)
        ax.set_xlabel('일관성 점수 (높을수록 일관적)')
        ax.set_title('Task별 Trajectory 일관성')
        ax.set_xlim(0, 1)
        ax.grid(axis='x', alpha=0.3)
    
    # 액션 분포 (전체)
    ax = axes[1, 1]
    all_actions = np.concatenate([ep['actions'] for ep in episodes])
    ax.scatter(all_actions[:, 0], all_actions[:, 1], alpha=0.1, s=1)
    ax.set_xlabel('linear_x (전진/후진)')
    ax.set_ylabel('linear_y (좌/우)')
    ax.set_title('전체 액션 분포')
    ax.grid(alpha=0.3)
    ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
    ax.axvline(0, color='k', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    output_path = Path("h5_episode_analysis.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ 시각화 저장: {output_path}")
    
    # 2. Task별 Trajectory 시각화 (상위 5개 task)
    top_tasks = sorted(task_stats.items(), key=lambda x: x[1]['num_episodes'], reverse=True)[:5]
    
    if top_tasks:
        fig, axes = plt.subplots(len(top_tasks), 1, figsize=(15, 3*len(top_tasks)))
        if len(top_tasks) == 1:
            axes = [axes]
        
        for idx, (task_name, stats) in enumerate(top_tasks):
            ax = axes[idx]
            
            # 해당 task의 모든 trajectory 수집
            task_episodes = task_groups[task_name]
            all_trajs = []
            for ep in task_episodes:
                trajs = analyze_trajectory(ep['actions'], window_size=18)
                if trajs:
                    all_trajs.extend(trajs)
            
            if all_trajs:
                # 가장 흔한 trajectory 시각화
                traj_strings = [trajectory_to_string(traj) for traj in all_trajs]
                from collections import Counter
                traj_counts = Counter(traj_strings)
                most_common_traj = traj_counts.most_common(1)[0][0]
                
                # 해당 trajectory 찾기
                for traj in all_trajs:
                    if trajectory_to_string(traj) == most_common_traj:
                        # Trajectory 플롯
                        ax.plot(traj[:, 0], label='linear_x', marker='o', markersize=3)
                        ax.plot(traj[:, 1], label='linear_y', marker='s', markersize=3)
                        ax.set_title(f"{task_name}\n가장 흔한 Trajectory: '{most_common_traj}' ({traj_counts[most_common_traj]}회)")
                        ax.set_xlabel('프레임 (0-17)')
                        ax.set_ylabel('액션 값')
                        ax.legend()
                        ax.grid(alpha=0.3)
                        break
        
        plt.tight_layout()
        output_path2 = Path("h5_trajectory_analysis.png")
        plt.savefig(output_path2, dpi=150, bbox_inches='tight')
        print(f"✅ Trajectory 시각화 저장: {output_path2}")
    
    # JSON 통계 저장
    stats_output = {
        'analysis_date': datetime.now().isoformat(),
        'total_episodes': len(episodes),
        'total_frames': sum(all_frame_counts),
        'avg_frames_per_episode': float(np.mean(all_frame_counts)),
        'num_tasks': len(task_groups),
        'task_stats': {k: {
            'num_episodes': v['num_episodes'],
            'num_trajectories': v['num_trajectories'],
            'num_unique_trajectories': v['num_unique_trajectories'],
            'consistency': float(v['consistency']),
            'most_common_trajectories': [{'trajectory': t[0], 'count': t[1]} for t in v['most_common']]
        } for k, v in task_stats.items()}
    }
    
    stats_path = Path("h5_episode_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats_output, f, indent=2)
    print(f"✅ 통계 저장: {stats_path}")
    
    print("\n" + "=" * 60)
    print("분석 완료!")
    print("=" * 60)

if __name__ == "__main__":
    main()

