#!/usr/bin/env python3
"""
전체 500개 에피소드 색상 스캔
Red 채널 결핍 및 색상 이상 탐지
"""

import cv2
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm


def analyze_episode_color(episode_dir: Path):
    """에피소드 색상 분석 (정상 프레임만)"""
    
    # 알려진 오류 프레임 제외
    error_frames = {'frame_0006.png', 'frame_0007.png'}
    
    frames = sorted([f for f in episode_dir.glob("frame_*.png") 
                    if f.name not in error_frames])[:16]
    
    if not frames:
        return None
    
    rgb_vals = []
    
    for frame_path in frames:
        img = cv2.imread(str(frame_path))
        if img is None:
            continue
        
        b, g, r = cv2.split(img)
        rgb_vals.append({
            'r': float(np.mean(r)),
            'g': float(np.mean(g)),
            'b': float(np.mean(b))
        })
    
    if not rgb_vals:
        return None
    
    avg_r = np.mean([v['r'] for v in rgb_vals])
    avg_g = np.mean([v['g'] for v in rgb_vals])
    avg_b = np.mean([v['b'] for v in rgb_vals])
    
    return {
        'episode_name': episode_dir.name,
        'avg_rgb': {
            'r': avg_r,
            'g': avg_g,
            'b': avg_b
        },
        'ratios': {
            'rg': avg_r / (avg_g + 1e-6),
            'rb': avg_r / (avg_b + 1e-6),
            'gb': avg_g / (avg_b + 1e-6)
        },
        'red_deficiency': (1 - avg_r / (avg_g + 1e-6)) * 100,  # Red가 Green보다 얼마나 적은가
        'blue_excess': (avg_b - avg_r),  # Blue가 Red보다 얼마나 많은가
    }


def scan_all_episodes():
    """전체 500개 에피소드 스캔"""
    
    base_dir = Path(__file__).parent / 'extracted_images'
    all_episodes = sorted([d for d in base_dir.iterdir() if d.is_dir()])
    
    print("🔍 전체 에피소드 색상 스캔 시작")
    print(f"   총 에피소드: {len(all_episodes)}")
    print("=" * 80)
    
    results = []
    
    for ep_dir in tqdm(all_episodes, desc="색상 분석"):
        result = analyze_episode_color(ep_dir)
        if result:
            results.append(result)
    
    # 통계 계산
    rg_ratios = [r['ratios']['rg'] for r in results]
    red_deficiencies = [r['red_deficiency'] for r in results]
    blue_excesses = [r['blue_excess'] for r in results]
    
    mean_rg = np.mean(rg_ratios)
    std_rg = np.std(rg_ratios)
    
    mean_red_def = np.mean(red_deficiencies)
    std_red_def = np.std(red_deficiencies)
    
    print("\n" + "=" * 80)
    print("📊 전체 통계")
    print("=" * 80)
    print(f"R/G 비율:")
    print(f"  평균: {mean_rg:.3f}")
    print(f"  표준편차: {std_rg:.3f}")
    print(f"  범위: {np.min(rg_ratios):.3f} ~ {np.max(rg_ratios):.3f}")
    print(f"\nRed 부족도:")
    print(f"  평균: {mean_red_def:.1f}%")
    print(f"  범위: {np.min(red_deficiencies):.1f}% ~ {np.max(red_deficiencies):.1f}%")
    
    # 이상치 탐지 (평균 - 2σ 보다 낮은 R/G 비율)
    threshold_rg = mean_rg - 2 * std_rg
    
    print(f"\n🔍 색상 이상 탐지 기준:")
    print(f"  R/G < {threshold_rg:.3f} (평균 - 2σ)")
    
    problematic = [r for r in results if r['ratios']['rg'] < threshold_rg]
    
    print(f"\n⚠️  이상 에피소드: {len(problematic)}개 발견")
    
    if problematic:
        # Red 부족도 순으로 정렬
        problematic.sort(key=lambda x: x['red_deficiency'], reverse=True)
        
        print("\n이상 에피소드 목록 (Red 부족도 순):")
        for i, ep in enumerate(problematic[:20], 1):  # 상위 20개만 표시
            name = ep['episode_name']
            rg = ep['ratios']['rg']
            red_def = ep['red_deficiency']
            
            print(f"{i:2d}. {name[-50:]}")
            print(f"    R/G: {rg:.3f} | Red 부족: {red_def:.1f}%")
    
    # 결과 저장
    output_dir = Path(__file__).parent / 'color_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / 'all_episodes_color_scan.json'
    with open(output_file, 'w') as f:
        json.dump({
            'total_episodes': len(results),
            'statistics': {
                'mean_rg': float(mean_rg),
                'std_rg': float(std_rg),
                'threshold_rg': float(threshold_rg),
                'problematic_count': len(problematic)
            },
            'problematic_episodes': problematic,
            'all_results': results
        }, f, indent=2)
    
    print(f"\n💾 결과 저장: {output_file}")
    
    return results, problematic


if __name__ == '__main__':
    results, problematic = scan_all_episodes()
    print("\n✅ 전체 스캔 완료!")
