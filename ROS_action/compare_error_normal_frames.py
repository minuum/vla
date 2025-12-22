#!/usr/bin/env python3
"""
오류 프레임 vs 정상 프레임 직접 비교
프레임 레벨 상세 색상 분석
"""

import cv2
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt


def analyze_frame_color(frame_path: Path):
    """단일 프레임 색상 상세 분석"""
    img = cv2.imread(str(frame_path))
    if img is None:
        return None
    
    # BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # BGR to HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # RGB 채널
    b, g, r = cv2.split(img)
    
    # HSV 채널
    h, s, v = cv2.split(img_hsv)
    
    return {
        'frame_name': frame_path.name,
        'rgb': {
            'mean_r': float(np.mean(r)),
            'mean_g': float(np.mean(g)),
            'mean_b': float(np.mean(b)),
            'std_r': float(np.std(r)),
            'std_g': float(np.std(g)),
            'std_b': float(np.std(b)),
        },
        'hsv': {
            'mean_h': float(np.mean(h)),
            'mean_s': float(np.mean(s)),
            'mean_v': float(np.mean(v)),
            'std_s': float(np.std(s)),
        },
        'ratios': {
            'rg': float(np.mean(r) / (np.mean(g) + 1e-6)),
            'rb': float(np.mean(r) / (np.mean(b) + 1e-6)),
            'gb': float(np.mean(g) / (np.mean(b) + 1e-6)),
        },
        'histograms': {
            'r': np.histogram(r, bins=256, range=(0, 256))[0].tolist(),
            'g': np.histogram(g, bins=256, range=(0, 256))[0].tolist(),
            'b': np.histogram(b, bins=256, range=(0, 256))[0].tolist(),
            's': np.histogram(s, bins=256, range=(0, 256))[0].tolist(),
        }
    }


def compare_error_vs_normal():
    """오류 프레임 vs 정상 프레임 직접 비교"""
    
    base_dir = Path(__file__).parent / 'extracted_images'
    
    # Episode 1
    ep1_dir = base_dir / 'episode_20251204_000842_1box_hori_right_core_medium'
    
    frames_to_analyze = {
        'ep1_normal_before': ep1_dir / 'frame_0005.png',
        'ep1_error_1': ep1_dir / 'frame_0006.png',
        'ep1_error_2': ep1_dir / 'frame_0007.png',
        'ep1_normal_after': ep1_dir / 'frame_0008.png',
    }
    
    # Episode 2
    ep2_dir = base_dir / 'episode_20251204_013302_1box_hori_right_core_medium'
    
    ep2_frames = {
        'ep2_normal_before': ep2_dir / 'frame_0006.png',
        'ep2_error': ep2_dir / 'frame_0007.png',
        'ep2_normal_after': ep2_dir / 'frame_0008.png',
    }
    
    frames_to_analyze.update(ep2_frames)
    
    print("🔍 오류 프레임 vs 정상 프레임 상세 비교")
    print("=" * 80)
    
    results = {}
    for name, path in frames_to_analyze.items():
        if path.exists():
            result = analyze_frame_color(path)
            if result:
                results[name] = result
                
                status = "❌ ERROR" if 'error' in name else "✅ NORMAL"
                print(f"\n{status} {name}")
                print(f"  RGB: R={result['rgb']['mean_r']:.1f} G={result['rgb']['mean_g']:.1f} B={result['rgb']['mean_b']:.1f}")
                print(f"  HSV: H={result['hsv']['mean_h']:.1f} S={result['hsv']['mean_s']:.1f} V={result['hsv']['mean_v']:.1f}")
                print(f"  Ratios: R/G={result['ratios']['rg']:.3f} R/B={result['ratios']['rb']:.3f} G/B={result['ratios']['gb']:.3f}")
    
    # 비교 분석
    print("\n" + "=" * 80)
    print("📊 비교 분석")
    print("=" * 80)
    
    # Episode 1 분석
    if all(k in results for k in ['ep1_normal_before', 'ep1_error_1', 'ep1_error_2', 'ep1_normal_after']):
        print("\nEpisode 1 시퀀스 분석:")
        
        normal = results['ep1_normal_before']
        error1 = results['ep1_error_1']
        error2 = results['ep1_error_2']
        
        print(f"\n채도 변화:")
        print(f"  정상(0005): {normal['hsv']['mean_s']:.1f}")
        print(f"  오류(0006): {error1['hsv']['mean_s']:.1f} (Δ{error1['hsv']['mean_s'] - normal['hsv']['mean_s']:+.1f})")
        print(f"  오류(0007): {error2['hsv']['mean_s']:.1f} (Δ{error2['hsv']['mean_s'] - normal['hsv']['mean_s']:+.1f})")
        
        print(f"\nRGB 밸런스 변화:")
        print(f"  정상 R/G: {normal['ratios']['rg']:.3f}")
        print(f"  오류1 R/G: {error1['ratios']['rg']:.3f} (Δ{error1['ratios']['rg'] - normal['ratios']['rg']:+.3f})")
        print(f"  오류2 R/G: {error2['ratios']['rg']:.3f} (Δ{error2['ratios']['rg'] - normal['ratios']['rg']:+.3f})")
        
        print(f"\n  정상 G/B: {normal['ratios']['gb']:.3f}")
        print(f"  오류1 G/B: {error1['ratios']['gb']:.3f} (Δ{error1['ratios']['gb'] - normal['ratios']['gb']:+.3f})")
        print(f"  오류2 G/B: {error2['ratios']['gb']:.3f} (Δ{error2['ratios']['gb'] - normal['ratios']['gb']:+.3f})")
    
    # 결과 저장
    output_dir = Path(__file__).parent / 'color_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / 'frame_level_color_comparison.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 결과 저장: {output_file}")
    
    # 시각화
    create_frame_comparison_plot(results, output_dir)
    
    return results


def create_frame_comparison_plot(results: dict, output_dir: Path):
    """프레임별 비교 시각화"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 순서 정의
    frame_order = [
        'ep1_normal_before',
        'ep1_error_1',
        'ep1_error_2',
        'ep1_normal_after',
        'ep2_normal_before',
        'ep2_error',
        'ep2_normal_after'
    ]
    
    # 실제 존재하는 프레임만 필터링
    ordered_results = [(name, results[name]) for name in frame_order if name in results]
    
    labels = [name.replace('_', '\n') for name, _ in ordered_results]
    colors = ['green' if 'normal' in name else 'red' for name, _ in ordered_results]
    
    # 1. RGB 채널 비교
    ax = axes[0, 0]
    r_vals = [r['rgb']['mean_r'] for _, r in ordered_results]
    g_vals = [r['rgb']['mean_g'] for _, r in ordered_results]
    b_vals = [r['rgb']['mean_b'] for _, r in ordered_results]
    
    x = np.arange(len(ordered_results))
    width = 0.25
    
    ax.bar(x - width, r_vals, width, label='Red', color='red', alpha=0.7)
    ax.bar(x, g_vals, width, label='Green', color='green', alpha=0.7)
    ax.bar(x + width, b_vals, width, label='Blue', color='blue', alpha=0.7)
    
    ax.set_ylabel('Mean Channel Value')
    ax.set_title('RGB Channels by Frame')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 채도 비교
    ax = axes[0, 1]
    saturations = [r['hsv']['mean_s'] for _, r in ordered_results]
    bars = ax.bar(range(len(saturations)), saturations, color=colors, alpha=0.7)
    
    ax.set_ylabel('Saturation')
    ax.set_title('Saturation by Frame')
    ax.set_xticks(range(len(ordered_results)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 3. RGB 비율
    ax = axes[1, 0]
    rg_ratios = [r['ratios']['rg'] for _, r in ordered_results]
    gb_ratios = [r['ratios']['gb'] for _, r in ordered_results]
    
    ax.plot(range(len(rg_ratios)), rg_ratios, 'o-', label='R/G', color='orange', linewidth=2)
    ax.plot(range(len(gb_ratios)), gb_ratios, 's-', label='G/B', color='purple', linewidth=2)
    
    ax.set_ylabel('Ratio')
    ax.set_title('RGB Channel Ratios')
    ax.set_xticks(range(len(ordered_results)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. 채도 히스토그램 (에피소드1 예시)
    ax = axes[1, 1]
    
    if 'ep1_normal_before' in results and 'ep1_error_1' in results:
        normal_hist = results['ep1_normal_before']['histograms']['s']
        error_hist = results['ep1_error_1']['histograms']['s']
        
        bins = np.arange(256)
        ax.plot(bins, normal_hist, label='Normal (frame_0005)', color='green', alpha=0.7)
        ax.plot(bins, error_hist, label='Error (frame_0006)', color='red', alpha=0.7)
        
        ax.set_xlabel('Saturation Value')
        ax.set_ylabel('Pixel Count')
        ax.set_title('Saturation Distribution (Episode 1)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'frame_level_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 그래프 저장: {output_path}")


if __name__ == '__main__':
    compare_error_vs_normal()
    print("\n✅ 프레임 레벨 분석 완료!")
