import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # GUI 없이 이미지만 저장
import matplotlib.pyplot as plt
import json
from pathlib import Path

def analyze_cup_dataset():
    print("=== 상세한 cup.npy 데이터셋 분석 ===")
    
    # 데이터 로드
    data = np.load('cup.npy', allow_pickle=True)
    print(f"데이터 타입: {type(data)}")
    print(f"전체 데이터 샘플 수: {len(data)}")
    
    # 첫 번째 샘플 분석
    sample = data[0]
    print(f"\n=== 첫 번째 샘플 구조 ===")
    print(f"샘플 키: {sample.keys()}")
    
    # Robot State 분석
    print(f"\n=== Robot State 분석 ===")
    robot_state = sample['robot_state']
    print(f"Robot state 형태: {robot_state.shape}")
    print(f"Robot state 데이터 타입: {robot_state.dtype}")
    print(f"Robot state 샘플:")
    print(robot_state[:5])  # 처음 5개 상태만 출력
    
    # Task 분석
    print(f"\n=== Task 분석 ===")
    task = sample['task']
    print(f"Task 형태: {task.shape}")
    print(f"Task 내용: {task[0][0]}")
    
    # Other 분석
    print(f"\n=== Other 데이터 분석 ===")
    other = sample['other']
    print(f"Other 키: {other.keys()}")
    
    # Hand Image 분석
    hand_image = other['hand_image']
    print(f"\nHand Image:")
    print(f"  형태: {hand_image.shape}")
    print(f"  데이터 타입: {hand_image.dtype}")
    print(f"  최솟값: {hand_image.min()}, 최댓값: {hand_image.max()}")
    
    # Third Person Image 분석
    third_person_image = other['third_person_image']
    print(f"\nThird Person Image:")
    print(f"  형태: {third_person_image.shape}")
    print(f"  데이터 타입: {third_person_image.dtype}")
    print(f"  최솟값: {third_person_image.min()}, 최댓값: {third_person_image.max()}")
    
    # 시간 시퀀스 이미지 시각화 (처음 10개 프레임)
    sample = data[0]  # 첫 번째 (유일한) 샘플
    hand_images = sample['other']['hand_image']
    third_images = sample['other']['third_person_image']
    
    num_frames_to_show = min(10, hand_images.shape[0])
    fig, axes = plt.subplots(2, num_frames_to_show, figsize=(20, 8))
    
    for i in range(num_frames_to_show):
        # Hand image 시각화
        hand_img = hand_images[i]
        axes[0, i].imshow(hand_img)
        axes[0, i].set_title(f'Hand Frame {i+1}')
        axes[0, i].axis('off')
        
        # Third person image 시각화 (RGBA -> RGB)
        third_img = third_images[i]
        if third_img.shape[-1] == 4:  # RGBA
            third_img_rgb = third_img[:,:,:3]  # RGB 채널만 사용
            # float32를 uint8로 정규화
            if third_img_rgb.dtype == np.float32:
                third_img_rgb = np.clip(third_img_rgb, 0, 255).astype(np.uint8)
        else:
            third_img_rgb = third_img
            
        axes[1, i].imshow(third_img_rgb)
        axes[1, i].set_title(f'3rd Person Frame {i+1}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('cup_dataset_images.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n이미지 시각화 저장: cup_dataset_images.png (처음 {num_frames_to_show}개 프레임)")
    
    # 특정 프레임들만 별도로 저장 (시작, 중간, 끝)
    key_frames = [0, hand_images.shape[0]//2, hand_images.shape[0]-1]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for idx, frame_num in enumerate(key_frames):
        # Hand image
        axes[0, idx].imshow(hand_images[frame_num])
        axes[0, idx].set_title(f'Hand Frame {frame_num+1}')
        axes[0, idx].axis('off')
        
        # Third person image
        third_img_rgb = third_images[frame_num][:,:,:3]
        if third_img_rgb.dtype == np.float32:
            third_img_rgb = np.clip(third_img_rgb, 0, 255).astype(np.uint8)
        axes[1, idx].imshow(third_img_rgb)
        axes[1, idx].set_title(f'3rd Person Frame {frame_num+1}')
        axes[1, idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('cup_key_frames.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"핵심 프레임 저장: cup_key_frames.png")
    
    # 로봇 상태를 CSV로 저장
    print(f"\n=== CSV 파일 생성 ===")
    
    sample = data[0]  # 첫 번째 (유일한) 샘플
    
    # Robot states CSV (108 타임스텝 x 15 상태 변수)
    robot_df = pd.DataFrame(sample['robot_state'], 
                          columns=[f'state_{j}' for j in range(sample['robot_state'].shape[1])])
    robot_df.index.name = 'timestep'
    robot_df.to_csv('robot_states_timeseries.csv', index=True)
    print(f"Robot states 저장: robot_states_timeseries.csv ({robot_df.shape})")
    
    # Tasks CSV
    task_df = pd.DataFrame({'task': sample['task'].flatten()})
    task_df.index.name = 'timestep'
    task_df.to_csv('tasks_timeseries.csv', index=True)
    print(f"Tasks 저장: tasks_timeseries.csv ({task_df.shape})")
    
    # Action states CSV (if exists)
    if 'action' in sample:
        action_df = pd.DataFrame(sample['action'], 
                               columns=[f'action_{j}' for j in range(sample['action'].shape[1])])
        action_df.index.name = 'timestep'
        action_df.to_csv('actions_timeseries.csv', index=True)
        print(f"Actions 저장: actions_timeseries.csv ({action_df.shape})")
    
    # 로봇 상태 통계 분석
    robot_stats = robot_df.describe()
    robot_stats.to_csv('robot_states_statistics.csv')
    print(f"Robot states 통계 저장: robot_states_statistics.csv")
    
    # 전체 데이터셋 요약 JSON
    sample = data[0]
    summary = {
        'dataset_type': 'VLA_Robot_Demonstration',
        'total_samples': len(data),
        'timesteps_per_sample': sample['robot_state'].shape[0],
        'data_structure': {
            'robot_state': {
                'shape': list(sample['robot_state'].shape),
                'description': 'Robot joint positions and gripper state over time',
                'columns': [f'state_{i}' for i in range(sample['robot_state'].shape[1])]
            },
            'task': {
                'shape': list(sample['task'].shape),
                'description': 'Task instruction repeated for each timestep',
                'content': sample['task'][0][0]
            },
            'hand_image': {
                'shape': list(sample['other']['hand_image'].shape),
                'description': 'Hand/wrist camera images over time (RGB)',
                'format': 'uint8'
            },
            'third_person_image': {
                'shape': list(sample['other']['third_person_image'].shape),
                'description': 'Third-person view camera images over time (RGBA)',
                'format': 'float32'
            }
        },
        'statistics': {
            'robot_state': {
                'min': float(sample['robot_state'].min()),
                'max': float(sample['robot_state'].max()),
                'mean': float(sample['robot_state'].mean()),
                'std': float(sample['robot_state'].std())
            }
        }
    }
    
    if 'action' in sample:
        summary['data_structure']['action'] = {
            'shape': list(sample['action'].shape),
            'description': 'Robot actions taken at each timestep'
        }
    
    with open('cup_dataset_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"데이터셋 요약 저장: cup_dataset_summary.json")
    
    return data

if __name__ == "__main__":
    dataset = analyze_cup_dataset()