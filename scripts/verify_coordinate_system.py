#!/usr/bin/env python3
"""
좌표계 검증 스크립트
데이터셋의 linear_x, linear_y 값 분포를 분석하여 
전진/후진/좌/우 방향이 어떻게 정의되어 있는지 확인합니다.
"""

import h5py
import numpy as np
from pathlib import Path
import argparse
import sys

def analyze_episode(h5_path: Path):
    """단일 에피소드 파일 분석"""
    try:
        with h5py.File(h5_path, 'r') as f:
            actions = f['actions'][:]  # Shape: (N, 3) - [linear_x, linear_y, angular_z]
            
            # linear_x, linear_y만 추출
            linear_x = actions[:, 0]
            linear_y = actions[:, 1]
            
            return {
                'file': h5_path.name,
                'x_values': linear_x,
                'y_values': linear_y,
                'x_mean': np.mean(linear_x),
                'y_mean': np.mean(linear_y),
                'x_nonzero': linear_x[np.abs(linear_x) > 0.01],
                'y_nonzero': linear_y[np.abs(linear_y) > 0.01]
            }
    except Exception as e:
        print(f"❌ Error reading {h5_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Verify coordinate system of dataset')
    parser.add_argument('--dataset', type=str, default='/home/soda/vla/ROS_action/mobile_vla_dataset',
                        help='Path to dataset directory')
    parser.add_argument('--samples', type=int, default=5,
                        help='Number of episodes to sample')
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        sys.exit(1)
    
    # H5 파일 찾기
    h5_files = sorted(list(dataset_path.glob('episode_*.h5')))
    if not h5_files:
        print(f"❌ No episode files found in {dataset_path}")
        sys.exit(1)
    
    print(f"📂 Found {len(h5_files)} episodes")
    print(f"🔍 Analyzing {min(args.samples, len(h5_files))} samples...\n")
    
    # 샘플 분석
    results = []
    for h5_file in h5_files[:args.samples]:
        result = analyze_episode(h5_file)
        if result:
            results.append(result)
    
    if not results:
        print("❌ No valid results")
        sys.exit(1)
    
    # 통계 집계
    print("="*70)
    print("📊 COORDINATE SYSTEM ANALYSIS")
    print("="*70)
    
    all_x_nonzero = np.concatenate([r['x_nonzero'] for r in results if len(r['x_nonzero']) > 0])
    all_y_nonzero = np.concatenate([r['y_nonzero'] for r in results if len(r['y_nonzero']) > 0])
    
    print("\n[Linear X Statistics]")
    print(f"  Min: {np.min(all_x_nonzero):.4f}")
    print(f"  Max: {np.max(all_x_nonzero):.4f}")
    print(f"  Mean: {np.mean(all_x_nonzero):.4f}")
    print(f"  Positive count: {np.sum(all_x_nonzero > 0)} / {len(all_x_nonzero)}")
    print(f"  Negative count: {np.sum(all_x_nonzero < 0)} / {len(all_x_nonzero)}")
    
    print("\n[Linear Y Statistics]")
    print(f"  Min: {np.min(all_y_nonzero):.4f}")
    print(f"  Max: {np.max(all_y_nonzero):.4f}")
    print(f"  Mean: {np.mean(all_y_nonzero):.4f}")
    print(f"  Positive count: {np.sum(all_y_nonzero > 0)} / {len(all_y_nonzero)}")
    print(f"  Negative count: {np.sum(all_y_nonzero < 0)} / {len(all_y_nonzero)}")
    
    # 샘플별 상세 출력
    print("\n" + "="*70)
    print("📋 SAMPLE DETAILS")
    print("="*70)
    for result in results:
        print(f"\n{result['file']}:")
        print(f"  X mean: {result['x_mean']:+.4f}")
        print(f"  Y mean: {result['y_mean']:+.4f}")
        if len(result['x_nonzero']) > 0:
            print(f"  X non-zero: {result['x_nonzero'][:5].tolist()}")  # 처음 5개만
        if len(result['y_nonzero']) > 0:
            print(f"  Y non-zero: {result['y_nonzero'][:5].tolist()}")
    
    # 결론
    print("\n" + "="*70)
    print("🎯 CONCLUSION")
    print("="*70)
    
    x_positive_ratio = np.sum(all_x_nonzero > 0) / len(all_x_nonzero)
    y_positive_ratio = np.sum(all_y_nonzero > 0) / len(all_y_nonzero)
    
    print(f"\nX-axis: {x_positive_ratio*100:.1f}% positive")
    if x_positive_ratio > 0.7:
        print("  ✅ X > 0 dominates → Likely X=FORWARD")
    elif x_positive_ratio < 0.3:
        print("  ⚠️ X < 0 dominates → Likely X=BACKWARD (or inverted)")
    else:
        print("  ❓ Mixed values")
    
    print(f"\nY-axis: {y_positive_ratio*100:.1f}% positive")
    if y_positive_ratio > 0.7:
        print("  ✅ Y > 0 dominates → Likely Y=LEFT")
    elif y_positive_ratio < 0.3:
        print("  ⚠️ Y < 0 dominates → Likely Y=RIGHT (or inverted)")
    else:
        print("  ❓ Mixed values")
    
    # 권장 사항
    print("\n" + "="*70)
    print("💡 RECOMMENDATION")
    print("="*70)
    
    if x_positive_ratio > 0.7:
        print("✅ X-axis seems correct (Forward = +)")
    else:
        print("⚠️ Consider inverting X-axis in inference node")
    
    if y_positive_ratio > 0.7:
        print("✅ Y-axis seems correct (Left = +)")
    else:
        print("⚠️ Consider inverting Y-axis in inference node")

if __name__ == "__main__":
    main()
