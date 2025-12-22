#!/usr/bin/env python3
"""
색상 분석 도구
RGB 채널, HSV 분석을 통해 "물빠진" 색상 감지
"""

import cv2
import numpy as np
from pathlib import Path
import json
from typing import Dict
import matplotlib.pyplot as plt


def analyze_color_detailed(frame_path: Path) -> Dict:
    """프레임의 색상 상세 분석"""
    img = cv2.imread(str(frame_path))
    if img is None:
        return {'error': 'Failed to load'}
    
    # BGR to RGB for analysis
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # BGR to HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # RGB 채널 분석
    b, g, r = cv2.split(img)
    
    stats = {
        'frame_name': frame_path.name,
        
        # RGB 채널 평균
        'mean_red': float(np.mean(r)),
        'mean_green': float(np.mean(g)),
        'mean_blue': float(np.mean(b)),
        
        # RGB 채널 표준편차
        'std_red': float(np.std(r)),
        'std_green': float(np.std(g)),
        'std_blue': float(np.std(b)),
        
        # HSV 분석
        'mean_hue': float(np.mean(img_hsv[:,:,0])),  # 색조
        'mean_saturation': float(np.mean(img_hsv[:,:,1])),  # 채도
        'mean_value': float(np.mean(img_hsv[:,:,2])),  # 명도
        
        'std_saturation': float(np.std(img_hsv[:,:,1])),
        
        # 색상 균형 (R/G, R/B 비율)
        'rg_ratio': float(np.mean(r) / (np.mean(g) + 1e-6)),
        'rb_ratio': float(np.mean(r) / (np.mean(b) + 1e-6)),
        'gb_ratio': float(np.mean(g) / (np.mean(b) + 1e-6)),
        
        # 저채도 픽셀 비율 (채도 < 30)
        'low_saturation_ratio': float(np.sum(img_hsv[:,:,1] < 30) / img_hsv[:,:,1].size),
        
        # 회색조에 가까운 픽셀 비율
        'grayish_ratio': float(np.sum(img_hsv[:,:,1] < 20) / img_hsv[:,:,1].size),
    }
    
    return stats


def compare_color_between_episodes(episode_dirs: list, output_dir: Path):
    """여러 에피소드 간 색상 비교"""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🎨 에피소드 간 색상 비교 분석")
    print("=" * 80)
    
    all_episode_stats = []
    
    for ep_dir in episode_dirs:
        print(f"\n📂 {ep_dir.name}")
        
        frames = sorted(ep_dir.glob("frame_*.png"))
        frame_stats = []
        
        for frame_path in frames[:18]:  # 18프레임만
            stats = analyze_color_detailed(frame_path)
            if 'error' not in stats:
                frame_stats.append(stats)
        
        # 에피소드 평균 통계
        episode_avg = {
            'episode_name': ep_dir.name,
            'frame_count': len(frame_stats),
            'avg_saturation': np.mean([s['mean_saturation'] for s in frame_stats]),
            'avg_hue': np.mean([s['mean_hue'] for s in frame_stats]),
            'avg_value': np.mean([s['mean_value'] for s in frame_stats]),
            'avg_red': np.mean([s['mean_red'] for s in frame_stats]),
            'avg_green': np.mean([s['mean_green'] for s in frame_stats]),
            'avg_blue': np.mean([s['mean_blue'] for s in frame_stats]),
            'avg_low_sat_ratio': np.mean([s['low_saturation_ratio'] for s in frame_stats]),
            'avg_grayish_ratio': np.mean([s['grayish_ratio'] for s in frame_stats]),
            'frames': frame_stats
        }
        
        all_episode_stats.append(episode_avg)
        
        print(f"  평균 채도: {episode_avg['avg_saturation']:.1f}")
        print(f"  평균 명도: {episode_avg['avg_value']:.1f}")
        print(f"  저채도 픽셀: {episode_avg['avg_low_sat_ratio']*100:.1f}%")
        print(f"  회색조 픽셀: {episode_avg['avg_grayish_ratio']*100:.1f}%")
    
    # 비교 분석
    print("\n" + "=" * 80)
    print("📊 에피소드 비교")
    print("=" * 80)
    
    saturations = [e['avg_saturation'] for e in all_episode_stats]
    
    print(f"\n채도 (Saturation) 분석:")
    print(f"  평균: {np.mean(saturations):.1f}")
    print(f"  표준편차: {np.std(saturations):.1f}")
    print(f"  범위: {np.min(saturations):.1f} ~ {np.max(saturations):.1f}")
    
    # 이상치 찾기
    mean_sat = np.mean(saturations)
    std_sat = np.std(saturations)
    
    print(f"\n🔍 채도 이상치 감지 (평균 ± 2*표준편차):")
    for ep_stat in all_episode_stats:
        sat = ep_stat['avg_saturation']
        if abs(sat - mean_sat) > 2 * std_sat:
            print(f"  ⚠️  {ep_stat['episode_name'][-40:]}")
            print(f"     채도: {sat:.1f} (평균: {mean_sat:.1f})")
        
    # JSON 저장
    output_file = output_dir / 'color_comparison.json'
    with open(output_file, 'w') as f:
        json.dump({
            'episodes': all_episode_stats,
            'comparison': {
                'mean_saturation': float(np.mean(saturations)),
                'std_saturation': float(np.std(saturations)),
                'min_saturation': float(np.min(saturations)),
                'max_saturation': float(np.max(saturations))
            }
        }, f, indent=2)
    
    print(f"\n💾 결과 저장: {output_file}")
    
    # 시각화
    create_color_comparison_plots(all_episode_stats, output_dir)
    
    return all_episode_stats


def create_color_comparison_plots(episode_stats: list, output_dir: Path):
    """색상 비교 시각화"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    episode_names = [e['episode_name'][-40:] for e in episode_stats]
    
    # 1. 채도 비교
    ax = axes[0, 0]
    saturations = [e['avg_saturation'] for e in episode_stats]
    bars = ax.bar(range(len(saturations)), saturations)
    
    # 이상치 색상 표시
    mean_sat = np.mean(saturations)
    std_sat = np.std(saturations)
    for i, sat in enumerate(saturations):
        if abs(sat - mean_sat) > 2 * std_sat:
            bars[i].set_color('red')
    
    ax.axhline(mean_sat, color='blue', linestyle='--', label=f'Mean: {mean_sat:.1f}')
    ax.set_ylabel('Saturation')
    ax.set_title('Average Saturation by Episode')
    ax.legend()
    ax.set_xticks(range(len(episode_names)))
    ax.set_xticklabels(episode_names, rotation=45, ha='right', fontsize=6)
    
    # 2. RGB 채널 비교
    ax = axes[0, 1]
    reds = [e['avg_red'] for e in episode_stats]
    greens = [e['avg_green'] for e in episode_stats]
    blues = [e['avg_blue'] for e in episode_stats]
    
    x = np.arange(len(episode_stats))
    width = 0.25
    ax.bar(x - width, reds, width, label='Red', color='red', alpha=0.7)
    ax.bar(x, greens, width, label='Green', color='green', alpha=0.7)
    ax.bar(x + width, blues, width, label='Blue', color='blue', alpha=0.7)
    
    ax.set_ylabel('Mean Channel Value')
    ax.set_title('RGB Channel Comparison')
    ax.legend()
    ax.set_xticks(x)
    ax.set_xticklabels(episode_names, rotation=45, ha='right', fontsize=6)
    
    # 3. 저채도 픽셀 비율
    ax = axes[1, 0]
    low_sat_ratios = [e['avg_low_sat_ratio'] * 100 for e in episode_stats]
    ax.bar(range(len(low_sat_ratios)), low_sat_ratios)
    ax.set_ylabel('Low Saturation Pixels (%)')
    ax.set_title('Desaturated Pixel Ratio')
    ax.set_xticks(range(len(episode_names)))
    ax.set_xticklabels(episode_names, rotation=45, ha='right', fontsize=6)
    
    # 4. 채도 vs 명도 산점도
    ax = axes[1, 1]
    saturations = [e['avg_saturation'] for e in episode_stats]
    values = [e['avg_value'] for e in episode_stats]
    
    for i, (sat, val) in enumerate(zip(saturations, values)):
        # 이상치는 빨간색으로
        if abs(sat - mean_sat) > 2 * std_sat:
            ax.scatter(sat, val, c='red', s=100, marker='x', linewidths=3)
            ax.annotate(f"Ep {i}", (sat, val), fontsize=8)
        else:
            ax.scatter(sat, val, c='blue', alpha=0.6)
    
    ax.set_xlabel('Saturation')
    ax.set_ylabel('Value (Brightness)')
    ax.set_title('Saturation vs Brightness')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'color_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 그래프 저장: {output_path}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='에피소드 간 색상 비교 분석')
    parser.add_argument(
        '--extracted-dir',
        type=Path,
        default=Path(__file__).parent / 'extracted_images',
        help='추출된 이미지 디렉토리'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=20,
        help='분석할 샘플 에피소드 수'
    )
    parser.add_argument(
        '--include-problematic',
        action='store_true',
        default=True,
        help='오류 에피소드 포함'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path(__file__).parent / 'color_analysis',
        help='출력 디렉토리'
    )
    
    args = parser.parse_args()
    
    # 에피소드 선택
    all_episodes = sorted([d for d in args.extracted_dir.iterdir() if d.is_dir()])
    
    # 오류 에피소드
    problematic = [
        'episode_20251204_000842_1box_hori_right_core_medium',
        'episode_20251204_013302_1box_hori_right_core_medium'
    ]
    
    # 오류 에피소드 + 랜덤 샘플
    selected = []
    
    if args.include_problematic:
        for prob in problematic:
            ep_dir = args.extracted_dir / prob
            if ep_dir.exists():
                selected.append(ep_dir)
    
    # 나머지는 랜덤 샘플
    import random
    other_episodes = [e for e in all_episodes if e.name not in problematic]
    selected.extend(random.sample(other_episodes, min(args.sample_size - len(selected), len(other_episodes))))
    
    print(f"총 {len(selected)}개 에피소드 분석")
    
    compare_color_between_episodes(selected, args.output_dir)
    
    print("\n✅ 색상 분석 완료!")
