"""
Episode Demo Script
실제 로봇 배포 전 시뮬레이션 / 궤적 테스트

목적:
- 2~4개 에피소드로 모델 성능 확인
- 궤적 시각화
- 평균/표준편차 계산
"""

import h5py
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Dict
import cv2
from PIL import Image


class EpisodeDemo:
    """에피소드 데모 및 궤적 분석"""
    
    def __init__(self, data_dir: str):
        """
        Args:
            data_dir: 데이터셋 디렉토리
        """
        self.data_dir = Path(data_dir)
        
    def load_episode(self, episode_path: Path) -> Dict:
        """
        에피소드 로드
        
        Returns:
            {
                'images': (T, H, W, 3),
                'actions': (T, 3),
                'name': str
            }
        """
        with h5py.File(episode_path, 'r') as f:
            return {
                'images': f['images'][:],
                'actions': f['actions'][:],
                'name': episode_path.stem
            }
    
    def compute_trajectory(self, actions: np.ndarray) -> Dict:
        """
        Action에서 궤적 계산
        
        Args:
            actions: (T, 3) [linear_x, linear_y, angular_z]
            
        Returns:
            {
                'positions': (T, 2) [x, y] 누적 위치,
                'velocities': (T, 2) [vx, vy],
                'stats': {...}
            }
        """
        linear_x = actions[:, 0]
        linear_y = actions[:, 1]
        
        # Compute cumulative position (simplified, dt=0.1s)
        dt = 0.1  # 10Hz
        positions = np.cumsum(np.stack([linear_x, linear_y], axis=1) * dt, axis=0)
        
        # Add origin
        positions = np.vstack([[0, 0], positions])
        
        velocities = np.stack([linear_x, linear_y], axis=1)
        
        # Stats
        stats = {
            'total_distance': np.linalg.norm(positions[-1]),
            'mean_linear_x': float(linear_x.mean()),
            'std_linear_x': float(linear_x.std()),
            'mean_linear_y': float(linear_y.mean()),
            'std_linear_y': float(linear_y.std()),
            'duration_seconds': len(actions) * dt
        }
        
        return {
            'positions': positions,
            'velocities': velocities,
            'stats': stats
        }
    
    def visualize_trajectory(
        self,
        episodes: List[Dict],
        output_path: str,
        title: str = "Episode Trajectories"
    ):
        """
        궤적 시각화
        
        Args:
            episodes: List of episode dicts with 'actions'
            output_path: 저장 경로
            title: 플롯 제목
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Compute trajectories
        trajectories = []
        for ep in episodes:
            traj = self.compute_trajectory(ep['actions'])
            trajectories.append(traj)
        
        # Plot 1: XY Trajectory
        colors = plt.cm.tab10(range(len(episodes)))
        
        for i, (ep, traj) in enumerate(zip(episodes, trajectories)):
            positions = traj['positions']
            label = f"{ep['name'][:30]}..."  # Truncate names
            
            ax1.plot(
                positions[:, 0],
                positions[:, 1],
                'o-',
                color=colors[i],
                label=label,
                markersize=3,
                alpha=0.7
            )
            
            # Mark start and end
            ax1.plot(positions[0, 0], positions[0, 1], 'go', markersize=10, label='Start' if i == 0 else '')
            ax1.plot(positions[-1, 0], positions[-1, 1], 'r*', markersize=15, label='End' if i == 0 else '')
        
        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Y Position (m)')
        ax1.set_title('XY Trajectories')
        ax1.legend(fontsize=8)
        ax1.grid(True)
        ax1.axis('equal')
        
        # Plot 2: Velocity over time
        for i, (ep, traj) in enumerate(zip(episodes, trajectories)):
            velocities = traj['velocities']
            time = np.arange(len(velocities)) * 0.1
            
            ax2.plot(time, velocities[:, 0], '-', color=colors[i], label=f'linear_x (ep {i+1})', linewidth=2)
            ax2.plot(time, velocities[:, 1], '--', color=colors[i], label=f'linear_y (ep {i+1})', linewidth=2)
        
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Velocity (m/s)')
        ax2.set_title('Velocities over Time')
        ax2.legend(fontsize=8)
        ax2.grid(True)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"📊 Trajectory saved to {output_path}")
        
        plt.close()
    
    def print_stats(self, episodes: List[Dict]):
        """통계 출력"""
        print("\n" + "="*60)
        print("EPISODE STATISTICS")
        print("="*60)
        
        for i, ep in enumerate(episodes):
            traj = self.compute_trajectory(ep['actions'])
            stats = traj['stats']
            
            print(f"\n📁 Episode {i+1}: {ep['name']}")
            print(f"  Duration: {stats['duration_seconds']:.1f}s")
            print(f"  Total distance: {stats['total_distance']:.3f}m")
            print(f"  Mean linear_x: {stats['mean_linear_x']:.3f} m/s (±{stats['std_linear_x']:.3f})")
            print(f"  Mean linear_y: {stats['mean_linear_y']:.3f} m/s (±{stats['std_linear_y']:.3f})")
        
        # Aggregate stats
        if len(episodes) > 1:
            all_stats = [self.compute_trajectory(ep['actions'])['stats'] for ep in episodes]
            
            print(f"\n" + "-"*60)
            print("AGGREGATE STATISTICS")
            print("-"*60)
            
            mean_distance = np.mean([s['total_distance'] for s in all_stats])
            std_distance = np.std([s['total_distance'] for s in all_stats])
            
            mean_duration = np.mean([s['duration_seconds'] for s in all_stats])
            
            print(f"  Episodes: {len(episodes)}")
            print(f"  Avg distance: {mean_distance:.3f}m (±{std_distance:.3f})")
            print(f"  Avg duration: {mean_duration:.1f}s")
        
        print("="*60 + "\n")
    
    def run_demo(
        self,
        episode_names: List[str],
        output_dir: str = "docs/episode_demo"
    ):
        """
        데모 실행
        
        Args:
            episode_names: 에피소드 파일 이름 리스트
            output_dir: 결과 저장 디렉토리
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load episodes
        print(f"🚀 Loading {len(episode_names)} episodes...")
        episodes = []
        
        for name in episode_names:
            episode_path = self.data_dir / name
            if not episode_path.exists():
                print(f"  ⚠️  {name} not found, skipping")
                continue
            
            ep = self.load_episode(episode_path)
            episodes.append(ep)
            print(f"  ✅ {name} ({len(ep['actions'])} frames)")
        
        if not episodes:
            print("❌ No valid episodes found!")
            return
        
        # Visualize
        print(f"\n📊 Visualizing trajectories...")
        self.visualize_trajectory(
            episodes,
            output_path=str(output_dir / "trajectories.png"),
            title=f"Episode Trajectories ({len(episodes)} episodes)"
        )
        
        # Print stats
        self.print_stats(episodes)
        
        print(f"\n✅ Demo complete! Results saved to {output_dir}")


def main():
    """Main demo script"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run episode demo")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/soda/25-1kp/vla/ROS_action/mobile_vla_dataset",
        help="Dataset directory"
    )
    parser.add_argument(
        "--episodes",
        nargs="+",
        help="Episode filenames (e.g., episode_20251203_*.h5)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="docs/episode_demo",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    # Default episodes (2 Left + 2 Right) if not specified
    if not args.episodes:
        # TODO: Select representative episodes
        args.episodes = [
            "episode_20251207_061643_1box_hori_left_core_medium.h5",
            "episode_20251207_061646_1box_hori_left_core_medium.h5",
            "episode_20251207_061651_1box_hori_right_core_medium.h5",
            "episode_20251207_061653_1box_hori_right_core_medium.h5",
        ]
        print(f"Using default episodes: {len(args.episodes)}")
    
    # Run demo
    demo = EpisodeDemo(data_dir=args.data_dir)
    demo.run_demo(
        episode_names=args.episodes,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()
