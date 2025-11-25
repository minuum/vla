"""
Mobile-VLA 데이터셋 궤적 분포 분석 스크립트

분석 항목:
1. 에피소드 수 및 길이 분포
2. 액션 분포 (linear_x, angular_z)
3. 궤적 패턴 (직진, 회전, 정지 등)
4. 속도 범위 및 통계
5. 데이터셋 다양성 메트릭
"""

import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib
matplotlib.use('Agg')  # GUI 없이 실행
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

class MobileVLADatasetAnalyzer:
    """Mobile-VLA 데이터셋 분석기"""
    
    def __init__(self, dataset_dir: str):
        self.dataset_dir = Path(dataset_dir)
        self.h5_files = sorted(list(self.dataset_dir.glob("*.h5")))
        print(f"📂 Found {len(self.h5_files)} H5 files")
        
        self.analysis_results = {
            'episodes': [],
            'actions': [],
            'statistics': {}
        }
    
    def analyze_single_episode(self, h5_path: Path) -> Dict:
        """단일 에피소드 분석"""
        with h5py.File(h5_path, 'r') as f:
            # 데이터 로드
            actions = f['actions'][:]  # (N, 2) - [linear_x, angular_z]
            
            # 이미지가 있다면 로드
            images = f['observations']['images'][:] if 'observations' in f and 'images' in f['observations'] else None
            
            episode_info = {
                'file': h5_path.name,
                'length': len(actions),
                'linear_x': actions[:, 0],
                'angular_z': actions[:, 1],
                'has_images': images is not None
            }
            
            # 통계 계산
            episode_info['stats'] = {
                'linear_mean': np.mean(actions[:, 0]),
                'linear_std': np.std(actions[:, 0]),
                'linear_min': np.min(actions[:, 0]),
                'linear_max': np.max(actions[:, 0]),
                'angular_mean': np.mean(actions[:, 1]),
                'angular_std': np.std(actions[:, 1]),
                'angular_min': np.min(actions[:, 1]),
                'angular_max': np.max(actions[:, 1]),
            }
            
            return episode_info
    
    def classify_action_type(self, linear_x: float, angular_z: float) -> str:
        """액션 타입 분류"""
        LINEAR_THRESHOLD = 0.1  # m/s
        ANGULAR_THRESHOLD = 0.2  # rad/s
        
        is_moving = abs(linear_x) > LINEAR_THRESHOLD
        is_turning = abs(angular_z) > ANGULAR_THRESHOLD
        
        if not is_moving and not is_turning:
            return 'STOP'
        elif is_moving and not is_turning:
            return 'FORWARD' if linear_x > 0 else 'BACKWARD'
        elif not is_moving and is_turning:
            return 'TURN_LEFT' if angular_z > 0 else 'TURN_RIGHT'
        else:
            # 동시 이동
            if angular_z > 0:
                return 'FORWARD_LEFT' if linear_x > 0 else 'BACKWARD_LEFT'
            else:
                return 'FORWARD_RIGHT' if linear_x > 0 else 'BACKWARD_RIGHT'
    
    def analyze_all_episodes(self):
        """전체 에피소드 분석"""
        print("\n🔍 Analyzing all episodes...")
        
        all_actions = []
        episode_lengths = []
        action_type_counts = {
            'STOP': 0,
            'FORWARD': 0,
            'BACKWARD': 0,
            'TURN_LEFT': 0,
            'TURN_RIGHT': 0,
            'FORWARD_LEFT': 0,
            'FORWARD_RIGHT': 0,
            'BACKWARD_LEFT': 0,
            'BACKWARD_RIGHT': 0
        }
        
        for h5_file in tqdm(self.h5_files, desc="Processing episodes"):
            try:
                ep_info = self.analyze_single_episode(h5_file)
                self.analysis_results['episodes'].append(ep_info)
                
                episode_lengths.append(ep_info['length'])
                
                # 액션 수집
                for linear_x, angular_z in zip(ep_info['linear_x'], ep_info['angular_z']):
                    all_actions.append([linear_x, angular_z])
                    action_type = self.classify_action_type(linear_x, angular_z)
                    action_type_counts[action_type] += 1
                
            except Exception as e:
                print(f"⚠️  Error processing {h5_file.name}: {e}")
        
        # 전체 통계
        all_actions = np.array(all_actions)
        
        self.analysis_results['statistics'] = {
            'total_episodes': len(self.h5_files),
            'total_timesteps': len(all_actions),
            'avg_episode_length': np.mean(episode_lengths),
            'std_episode_length': np.std(episode_lengths),
            'min_episode_length': np.min(episode_lengths),
            'max_episode_length': np.max(episode_lengths),
            
            'linear_x_mean': np.mean(all_actions[:, 0]),
            'linear_x_std': np.std(all_actions[:, 0]),
            'linear_x_min': np.min(all_actions[:, 0]),
            'linear_x_max': np.max(all_actions[:, 0]),
            
            'angular_z_mean': np.mean(all_actions[:, 1]),
            'angular_z_std': np.std(all_actions[:, 1]),
            'angular_z_min': np.min(all_actions[:, 1]),
            'angular_z_max': np.max(all_actions[:, 1]),
            
            'action_type_counts': action_type_counts
        }
        
        self.all_actions = all_actions
        self.episode_lengths = episode_lengths
    
    def generate_summary_table(self) -> pd.DataFrame:
        """요약 테이블 생성"""
        stats = self.analysis_results['statistics']
        
        summary_data = {
            '항목': [
                '총 에피소드 수',
                '총 타임스텝 수',
                '평균 에피소드 길이',
                '에피소드 길이 (최소/최대)',
                '',
                'Linear Velocity (m/s)',
                '  - 평균 ± 표준편차',
                '  - 범위 (최소/최대)',
                '',
                'Angular Velocity (rad/s)',
                '  - 평균 ± 표준편차',
                '  - 범위 (최소/최대)',
            ],
            '값': [
                f"{stats['total_episodes']}개",
                f"{stats['total_timesteps']:,}개",
                f"{stats['avg_episode_length']:.1f} ± {stats['std_episode_length']:.1f}",
                f"{stats['min_episode_length']} / {stats['max_episode_length']}",
                '',
                '',
                f"{stats['linear_x_mean']:.3f} ± {stats['linear_x_std']:.3f}",
                f"{stats['linear_x_min']:.3f} / {stats['linear_x_max']:.3f}",
                '',
                '',
                f"{stats['angular_z_mean']:.3f} ± {stats['angular_z_std']:.3f}",
                f"{stats['angular_z_min']:.3f} / {stats['angular_z_max']:.3f}",
            ]
        }
        
        return pd.DataFrame(summary_data)
    
    def generate_action_distribution_table(self) -> pd.DataFrame:
        """액션 분포 테이블 생성"""
        action_counts = self.analysis_results['statistics']['action_type_counts']
        total = sum(action_counts.values())
        
        action_data = {
            '액션 타입': [],
            '개수': [],
            '비율 (%)': [],
            '설명': []
        }
        
        action_descriptions = {
            'FORWARD': '직진',
            'BACKWARD': '후진',
            'TURN_LEFT': '제자리 좌회전',
            'TURN_RIGHT': '제자리 우회전',
            'FORWARD_LEFT': '전진 + 좌회전',
            'FORWARD_RIGHT': '전진 + 우회전',
            'BACKWARD_LEFT': '후진 + 좌회전',
            'BACKWARD_RIGHT': '후진 + 우회전',
            'STOP': '정지'
        }
        
        # 빈도순 정렬
        sorted_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)
        
        for action_type, count in sorted_actions:
            action_data['액션 타입'].append(action_type)
            action_data['개수'].append(f"{count:,}")
            action_data['비율 (%)'].append(f"{count/total*100:.1f}%")
            action_data['설명'].append(action_descriptions.get(action_type, '-'))
        
        return pd.DataFrame(action_data)
    
    def plot_distributions(self, save_path: str = None):
        """분포 시각화"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # 1. 에피소드 길이 분포
        axes[0, 0].hist(self.episode_lengths, bins=20, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Episode Length Distribution', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Episode Length (steps)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(np.mean(self.episode_lengths), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(self.episode_lengths):.1f}')
        axes[0, 0].legend()
        
        # 2. Linear Velocity 분포
        axes[0, 1].hist(self.all_actions[:, 0], bins=50, color='lightgreen', edgecolor='black')
        axes[0, 1].set_title('Linear Velocity Distribution', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Linear Velocity (m/s)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(0, color='black', linestyle='-', alpha=0.3)
        
        # 3. Angular Velocity 분포
        axes[0, 2].hist(self.all_actions[:, 1], bins=50, color='lightcoral', edgecolor='black')
        axes[0, 2].set_title('Angular Velocity Distribution', fontsize=14, fontweight='bold')
        axes[0, 2].set_xlabel('Angular Velocity (rad/s)')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].axvline(0, color='black', linestyle='-', alpha=0.3)
        
        # 4. 2D Joint Distribution
        axes[1, 0].scatter(self.all_actions[:, 0], self.all_actions[:, 1], 
                          alpha=0.1, s=1, color='navy')
        axes[1, 0].set_title('Action Space Distribution (2D)', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Linear Velocity (m/s)')
        axes[1, 0].set_ylabel('Angular Velocity (rad/s)')
        axes[1, 0].axhline(0, color='gray', linestyle='--', alpha=0.5)
        axes[1, 0].axvline(0, color='gray', linestyle='--', alpha=0.5)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Action Type Distribution (파이 차트)
        action_counts = self.analysis_results['statistics']['action_type_counts']
        # 5% 이상만 표시
        total = sum(action_counts.values())
        filtered_counts = {k: v for k, v in action_counts.items() if v/total > 0.05}
        other_count = sum(v for k, v in action_counts.items() if k not in filtered_counts)
        if other_count > 0:
            filtered_counts['OTHER'] = other_count
        
        axes[1, 1].pie(filtered_counts.values(), labels=filtered_counts.keys(), 
                      autopct='%1.1f%%', startangle=90)
        axes[1, 1].set_title('Action Type Distribution', fontsize=14, fontweight='bold')
        
        # 6. Velocity Magnitude Distribution
        velocity_magnitudes = np.sqrt(self.all_actions[:, 0]**2 + self.all_actions[:, 1]**2)
        axes[1, 2].hist(velocity_magnitudes, bins=50, color='orange', edgecolor='black')
        axes[1, 2].set_title('Velocity Magnitude Distribution', fontsize=14, fontweight='bold')
        axes[1, 2].set_xlabel('|v| = √(linear² + angular²)')
        axes[1, 2].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Plot saved to: {save_path}")
        
        return fig
    
    def export_analysis(self, output_dir: str = "analysis_results"):
        """분석 결과 저장"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 1. Summary Table
        summary_table = self.generate_summary_table()
        summary_table.to_csv(output_path / "dataset_summary.csv", index=False)
        print(f"✅ Summary table saved to: {output_path / 'dataset_summary.csv'}")
        
        # 2. Action Distribution Table
        action_table = self.generate_action_distribution_table()
        action_table.to_csv(output_path / "action_distribution.csv", index=False)
        print(f"✅ Action distribution saved to: {output_path / 'action_distribution.csv'}")
        
        # 3. Statistics JSON
        with open(output_path / "statistics.json", 'w') as f:
            json.dump(self.analysis_results['statistics'], f, indent=2)
        print(f"✅ Statistics JSON saved to: {output_path / 'statistics.json'}")
        
        # 4. Plots
        self.plot_distributions(save_path=output_path / "distributions.png")
        
        return summary_table, action_table

def main():
    """메인 실행 함수"""
    # 데이터셋 경로
    dataset_dir = "/Users/minu/dev/vla/ROS_action/mobile_vla_dataset"
    
    print("=" * 60)
    print("📊 Mobile-VLA Dataset Trajectory Analysis")
    print("=" * 60)
    
    # 분석기 초기화
    analyzer = MobileVLADatasetAnalyzer(dataset_dir)
    
    # 전체 분석 수행
    analyzer.analyze_all_episodes()
    
    # 결과 저장
    print("\n💾 Exporting analysis results...")
    summary_table, action_table = analyzer.export_analysis(
        output_dir="/Users/minu/dev/vla/docs/research/data_augmentation/analysis_results"
    )
    
    # 콘솔 출력
    print("\n" + "=" * 60)
    print("📋 DATASET SUMMARY")
    print("=" * 60)
    print(summary_table.to_string(index=False))
    
    print("\n" + "=" * 60)
    print("📊 ACTION TYPE DISTRIBUTION")
    print("=" * 60)
    print(action_table.to_string(index=False))
    
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()
