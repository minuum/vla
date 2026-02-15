#!/usr/bin/env python3
"""
에피소드 전체 색상 품질 비교
정상 프레임들끼리 비교하여 에피소드 레벨 색상 이상 감지
"""

import cv2
import numpy as np
from pathlib import Path
import json


def analyze_episode_normal_frames(episode_dir: Path):
    """에피소드의 정상 프레임들(오류 제외) 색상 분석"""
    
    # 오류 프레임 제외
    error_frames = {'frame_0006.png', 'frame_0007.png'}
    
    frames = sorted([f for f in episode_dir.glob("frame_*.png") 
                    if f.name not in error_frames])[:16]  # 정상 프레임만
    
    if not frames:
        return None
    
    rgb_vals = []
    hsv_vals = []
    
    for frame_path in frames:
        img = cv2.imread(str(frame_path))
        if img is None:
            continue
        
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        b, g, r = cv2.split(img)
        
        rgb_vals.append({
            'r': float(np.mean(r)),
            'g': float(np.mean(g)),
            'b': float(np.mean(b))
        })
        
        hsv_vals.append({
            'h': float(np.mean(img_hsv[:,:,0])),
            's': float(np.mean(img_hsv[:,:,1])),
            'v': float(np.mean(img_hsv[:,:,2]))
        })
    
    # 평균 계산
    return {
        'episode_name': episode_dir.name,
        'frame_count': len(rgb_vals),
        'avg_rgb': {
            'r': np.mean([v['r'] for v in rgb_vals]),
            'g': np.mean([v['g'] for v in rgb_vals]),
            'b': np.mean([v['b'] for v in rgb_vals])
        },
        'avg_hsv': {
            'h': np.mean([v['h'] for v in hsv_vals]),
            's': np.mean([v['s'] for v in hsv_vals]),
            'v': np.mean([v['v'] for v in hsv_vals])
        },
        'std_rgb': {
            'r': np.std([v['r'] for v in rgb_vals]),
            'g': np.std([v['g'] for v in rgb_vals]),
            'b': np.std([v['b'] for v in rgb_vals])
        },
        'ratios': {
            'rg': np.mean([v['r'] for v in rgb_vals]) / (np.mean([v['g'] for v in rgb_vals]) + 1e-6),
            'rb': np.mean([v['r'] for v in rgb_vals]) / (np.mean([v['b'] for v in rgb_vals]) + 1e-6),
            'gb': np.mean([v['g'] for v in rgb_vals]) / (np.mean([v['b'] for v in rgb_vals]) + 1e-6)
        }
    }


def compare_episodes():
    """문제 에피소드 vs 정상 에피소드 비교"""
    
    base_dir = Path(__file__).parent / 'extracted_images'
    
    # 문제 에피소드
    problematic = [
        'episode_20251204_000842_1box_hori_right_core_medium',
        'episode_20251204_013302_1box_hori_right_core_medium'
    ]
    
    # 모든 에피소드
    all_episodes = sorted([d for d in base_dir.iterdir() if d.is_dir()])
    
    # 샘플: 문제 에피소드 + 정상 에피소드 10개
    import random
    normal_sample = [e for e in all_episodes if e.name not in problematic]
    sample_episodes = [base_dir / p for p in problematic] + random.sample(normal_sample, 10)
    
    print("🔍 에피소드 전체 색상 품질 비교 (정상 프레임만)")
    print("=" * 80)
    
    results = []
    for ep_dir in sample_episodes:
        result = analyze_episode_normal_frames(ep_dir)
        if result:
            results.append(result)
            
            is_problematic = ep_dir.name in problematic
            marker = "⚠️ 문제" if is_problematic else "✅ 정상"
            
            print(f"\n{marker} {ep_dir.name[-40:]}")
            print(f"  RGB: R={result['avg_rgb']['r']:.1f} G={result['avg_rgb']['g']:.1f} B={result['avg_rgb']['b']:.1f}")
            print(f"  HSV: S={result['avg_hsv']['s']:.1f} V={result['avg_hsv']['v']:.1f}")
            print(f"  Ratios: R/G={result['ratios']['rg']:.3f} G/B={result['ratios']['gb']:.3f}")
    
    # 통계 비교
    print("\n" + "=" * 80)
    print("📊 통계 비교")
    print("=" * 80)
    
    # 문제 에피소드 vs 정상 에피소드
    prob_results = [r for r in results if any(p in r['episode_name'] for p in problematic)]
    normal_results = [r for r in results if not any(p in r['episode_name'] for p in problematic)]
    
    if prob_results and normal_results:
        print("\n문제 에피소드 (정상 프레임 평균):")
        prob_r = np.mean([r['avg_rgb']['r'] for r in prob_results])
        prob_g = np.mean([r['avg_rgb']['g'] for r in prob_results])
        prob_b = np.mean([r['avg_rgb']['b'] for r in prob_results])
        prob_s = np.mean([r['avg_hsv']['s'] for r in prob_results])
        
        print(f"  RGB: R={prob_r:.1f} G={prob_g:.1f} B={prob_b:.1f}")
        print(f"  Saturation: {prob_s:.1f}")
        print(f"  R/G: {prob_r/prob_g:.3f}")
        print(f"  G/B: {prob_g/prob_b:.3f}")
        
        print("\n정상 에피소드 (정상 프레임 평균):")
        norm_r = np.mean([r['avg_rgb']['r'] for r in normal_results])
        norm_g = np.mean([r['avg_rgb']['g'] for r in normal_results])
        norm_b = np.mean([r['avg_rgb']['b'] for r in normal_results])
        norm_s = np.mean([r['avg_hsv']['s'] for r in normal_results])
        
        print(f"  RGB: R={norm_r:.1f} G={norm_g:.1f} B={norm_b:.1f}")
        print(f"  Saturation: {norm_s:.1f}")
        print(f"  R/G: {norm_r/norm_g:.3f}")
        print(f"  G/B: {norm_g/norm_b:.3f}")
        
        print("\n차이 (문제 - 정상):")
        print(f"  ΔR: {prob_r - norm_r:+.1f}")
        print(f"  ΔG: {prob_g - norm_g:+.1f}")
        print(f"  ΔB: {prob_b - norm_b:+.1f}")
        print(f"  ΔSaturation: {prob_s - norm_s:+.1f}")
        print(f"  ΔR/G: {(prob_r/prob_g) - (norm_r/norm_g):+.3f}")
        print(f"  ΔG/B: {(prob_g/prob_b) - (norm_g/norm_b):+.3f}")
    
    # 저장
    output_dir = Path(__file__).parent / 'color_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / 'episode_level_comparison.json'
    with open(output_file, 'w') as f:
        json.dump({
            'problematic_episodes': prob_results,
            'normal_episodes': normal_results,
            'all_results': results
        }, f, indent=2)
    
    print(f"\n💾 결과 저장: {output_file}")
    
    return results


if __name__ == '__main__':
    compare_episodes()
    print("\n✅ 에피소드 레벨 비교 완료!")
