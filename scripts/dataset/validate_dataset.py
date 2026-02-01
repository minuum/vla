"""
Dataset Validation Script
HDF5 에피소드의 비디오 품질 및 액션 범위 검증

목적:
- 지지직 거리는 프레임 감지
- Action 범위 검증
- Episode 길이 확인
"""

import h5py
import numpy as np
from pathlib import Path
import cv2
from typing import Dict, List, Tuple
import json
from datetime import datetime


class DatasetValidator:
    """HDF5 데이터셋 검증기"""
    
    def __init__(self, data_dir: str, episode_pattern: str = "episode_*.h5"):
        """
        Args:
            data_dir: 데이터셋 디렉토리
            episode_pattern: 에피소드 파일 패턴
        """
        self.data_dir = Path(data_dir)
        self.episode_pattern = episode_pattern
        
        # 검증 기준
        self.FRAME_STD_THRESHOLD = 5.0  # 표준편차 너무 낮으면 손상
        self.FRAME_DIFF_THRESHOLD = 100.0  # 프레임 간 차이가 너무 크면 지지직
        self.MIN_EPISODE_LENGTH = 18  # 최소 프레임 수 (실제 데이터 기준)
        
        # Action 범위 (실제 데이터 기준)
        self.LINEAR_X_RANGE = (0.0, 1.5)  # linear_x 범위도 실제 데이터 맞춤
        self.LINEAR_Y_RANGE = (-1.5, 1.5)
        
    def find_episodes(self) -> List[Path]:
        """에피소드 파일 찾기"""
        episodes = sorted(self.data_dir.glob(self.episode_pattern))
        print(f"Found {len(episodes)} episodes")
        return episodes
    
    def check_frame_quality(self, images: np.ndarray) -> Dict:
        """
        프레임 품질 검사
        
        Args:
            images: (T, H, W, 3) RGB images
            
        Returns:
            {
                'corrupted_frames': List[int],  # 손상된 프레임 인덱스
                'noisy_frames': List[int],  # 지지직 프레임 인덱스
                'total_frames': int
            }
        """
        T = images.shape[0]
        corrupted_frames = []
        noisy_frames = []
        
        for t in range(T):
            frame = images[t]
            
            # 1. 그레이스케일 표준편차 체크 (너무 낮으면 손상)
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            std = np.std(gray)
            
            if std < self.FRAME_STD_THRESHOLD:
                corrupted_frames.append(t)
            
            # 2. 연속 프레임 간 차이 체크 (너무 크면 지지직)
            if t > 0:
                prev_frame = images[t-1]
                diff = np.abs(frame.astype(float) - prev_frame.astype(float)).mean()
                
                if diff > self.FRAME_DIFF_THRESHOLD:
                    noisy_frames.append(t)
        
        return {
            'corrupted_frames': corrupted_frames,
            'noisy_frames': noisy_frames,
            'total_frames': T
        }
    
    def check_action_range(self, actions: np.ndarray) -> Dict:
        """
        Action 범위 검증
        
        Args:
            actions: (T, 3) [linear_x, linear_y, angular_z]
            
        Returns:
            {
                'linear_x_valid': bool,
                'linear_y_valid': bool,
                'linear_x_range': (min, max),
                'linear_y_range': (min, max),
                'out_of_range_count': int
            }
        """
        linear_x = actions[:, 0]
        linear_y = actions[:, 1]
        
        # Check ranges
        linear_x_valid = np.all(
            (linear_x >= self.LINEAR_X_RANGE[0]) & 
            (linear_x <= self.LINEAR_X_RANGE[1])
        )
        
        linear_y_valid = np.all(
            (linear_y >= self.LINEAR_Y_RANGE[0]) & 
            (linear_y <= self.LINEAR_Y_RANGE[1])
        )
        
        out_of_range_count = (
            np.sum(~((linear_x >= self.LINEAR_X_RANGE[0]) & 
                    (linear_x <= self.LINEAR_X_RANGE[1]))) +
            np.sum(~((linear_y >= self.LINEAR_Y_RANGE[0]) & 
                    (linear_y <= self.LINEAR_Y_RANGE[1])))
        )
        
        return {
            'linear_x_valid': bool(linear_x_valid),
            'linear_y_valid': bool(linear_y_valid),
            'linear_x_range': (float(linear_x.min()), float(linear_x.max())),
            'linear_y_range': (float(linear_y.min()), float(linear_y.max())),
            'out_of_range_count': int(out_of_range_count)
        }
    
    def validate_episode(self, episode_path: Path) -> Dict:
        """
        단일 에피소드 검증
        
        Returns:
            {
                'episode_name': str,
                'valid': bool,
                'episode_length': int,
                'frame_quality': dict,
                'action_range': dict,
                'issues': List[str]
            }
        """
        issues = []
        
        try:
            with h5py.File(episode_path, 'r') as f:
                images = f['images'][:]  # (T, H, W, 3)
                actions = f['actions'][:]  # (T, 3)
                
                episode_length = images.shape[0]
                
                # 1. Length check
                if episode_length < self.MIN_EPISODE_LENGTH:
                    issues.append(f"Episode too short: {episode_length} < {self.MIN_EPISODE_LENGTH}")
                
                # 2. Frame quality
                frame_quality = self.check_frame_quality(images)
                if frame_quality['corrupted_frames']:
                    issues.append(f"Corrupted frames: {len(frame_quality['corrupted_frames'])}")
                if frame_quality['noisy_frames']:
                    issues.append(f"Noisy frames (지지직): {len(frame_quality['noisy_frames'])}")
                
                # 3. Action range
                action_range = self.check_action_range(actions)
                if not action_range['linear_x_valid']:
                    issues.append(f"linear_x out of range: {action_range['linear_x_range']}")
                if not action_range['linear_y_valid']:
                    issues.append(f"linear_y out of range: {action_range['linear_y_range']}")
                
                valid = len(issues) == 0
                
                return {
                    'episode_name': episode_path.name,
                    'valid': valid,
                    'episode_length': episode_length,
                    'frame_quality': frame_quality,
                    'action_range': action_range,
                    'issues': issues
                }
                
        except Exception as e:
            return {
                'episode_name': episode_path.name,
                'valid': False,
                'issues': [f"Failed to read: {str(e)}"]
            }
    
    def validate_all(self) -> Dict:
        """모든 에피소드 검증"""
        episodes = self.find_episodes()
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'total_episodes': len(episodes),
            'valid_episodes': 0,
            'invalid_episodes': 0,
            'episode_results': []
        }
        
        for episode_path in episodes:
            print(f"Validating {episode_path.name}...")
            
            result = self.validate_episode(episode_path)
            results['episode_results'].append(result)
            
            if result['valid']:
                results['valid_episodes'] += 1
                print(f"  ✅ Valid")
            else:
                results['invalid_episodes'] += 1
                print(f"  ❌ Invalid: {', '.join(result['issues'])}")
        
        return results
    
    def generate_report(self, results: Dict, output_path: str):
        """검증 리포트 생성"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Markdown report
        with open(output_path, 'w') as f:
            f.write("# Dataset Validation Report\n\n")
            f.write(f"**Generated**: {results['timestamp']}\n\n")
            f.write(f"**Total Episodes**: {results['total_episodes']}\n")
            f.write(f"**Valid Episodes**: {results['valid_episodes']}\n")
            f.write(f"**Invalid Episodes**: {results['invalid_episodes']}\n\n")
            
            # Summary
            f.write("## Summary\n\n")
            valid_rate = results['valid_episodes'] / results['total_episodes'] * 100
            f.write(f"- **Validation Rate**: {valid_rate:.1f}%\n\n")
            
            # Invalid episodes
            if results['invalid_episodes'] > 0:
                f.write("## Invalid Episodes\n\n")
                f.write("| Episode | Issues |\n")
                f.write("|---------|--------|\n")
                
                for result in results['episode_results']:
                    if not result['valid']:
                        issues_str = "<br>".join(result['issues'])
                        f.write(f"| {result['episode_name']} | {issues_str} |\n")
                
                f.write("\n")
            
            # Valid episodes
            f.write("## Valid Episodes\n\n")
            for result in results['episode_results']:
                if result['valid']:
                    f.write(f"- ✅ {result['episode_name']}\n")
        
        # JSON report
        json_path = output_path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📝 Report saved to:")
        print(f"  - {output_path}")
        print(f"  - {json_path}")


def main():
    """Main validation script"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate Mobile VLA dataset")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset",
        help="Dataset directory"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="episode_20251*.h5",
        help="Episode file pattern"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="docs/dataset_validation_report.md",
        help="Output report path"
    )
    
    args = parser.parse_args()
    
    # Create validator
    validator = DatasetValidator(
        data_dir=args.data_dir,
        episode_pattern=args.pattern
    )
    
    # Validate
    print("🔍 Starting dataset validation...")
    results = validator.validate_all()
    
    # Generate report
    validator.generate_report(results, args.output)
    
    print("\n✅ Validation complete!")
    print(f"Valid: {results['valid_episodes']}/{results['total_episodes']}")


if __name__ == "__main__":
    main()
