#!/usr/bin/env python3
"""
색상 보정 샘플 테스트 - Red 강도 증가
초록색 뚜껑이 제대로 보이도록 Red 채널 강화
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def method1_global_offset(img, offset=8):
    """Method 1: Red 채널에 고정 offset 추가 (증가)"""
    result = img.copy()
    result[:,:,2] = np.clip(result[:,:,2] + offset, 0, 255).astype(np.uint8)
    return result


def method2_white_balance_enhanced(img, red_boost=1.1):
    """Method 2: White Balance + Red boost"""
    b, g, r = cv2.split(img)
    
    r_mean = np.mean(r)
    g_mean = np.mean(g)
    b_mean = np.mean(b)
    
    avg_gray = (r_mean + g_mean + b_mean) / 3
    
    # Red에 추가 boost 적용
    r_corrected = np.clip(r * (avg_gray / (r_mean + 1e-6)) * red_boost, 0, 255).astype(np.uint8)
    g_corrected = np.clip(g * (avg_gray / (g_mean + 1e-6)), 0, 255).astype(np.uint8)
    b_corrected = np.clip(b * (avg_gray / (b_mean + 1e-6)), 0, 255).astype(np.uint8)
    
    return cv2.merge([b_corrected, g_corrected, r_corrected])


def method3_ratio_based_enhanced(img, target_rg=1.05):
    """Method 3: 목표 R/G 비율로 조정 (Red 우세하게)"""
    b, g, r = cv2.split(img)
    
    current_rg = np.mean(r) / (np.mean(g) + 1e-6)
    
    scale_r = target_rg / (current_rg + 1e-6)
    
    r_corrected = np.clip(r * scale_r, 0, 255).astype(np.uint8)
    
    return cv2.merge([b, g, r_corrected])


def analyze_rgb(img):
    """RGB 통계 분석"""
    b, g, r = cv2.split(img)
    
    return {
        'r_mean': float(np.mean(r)),
        'g_mean': float(np.mean(g)),
        'b_mean': float(np.mean(b)),
        'rg_ratio': float(np.mean(r) / (np.mean(g) + 1e-6)),
        'rb_ratio': float(np.mean(r) / (np.mean(b) + 1e-6)),
        'gb_ratio': float(np.mean(g) / (np.mean(b) + 1e-6))
    }


def test_sample_frames_enhanced():
    """샘플 프레임들로 보정 테스트 - Red 강화"""
    
    base_dir = Path(__file__).parent / 'extracted_images'
    
    # 샘플 에피소드 선택
    sample_episodes = [
        'episode_20251203_050009_1box_hori_left_core_medium',
        'episode_20251204_115106_1box_hori_right_core_medium',
        'episode_20251204_013302_1box_hori_right_core_medium',
    ]
    
    output_dir = Path(__file__).parent / 'correction_samples_enhanced'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🎨 색상 보정 샘플 테스트 (Red 강화)")
    print("=" * 80)
    
    all_results = []
    
    for ep_name in sample_episodes:
        ep_dir = base_dir / ep_name
        if not ep_dir.exists():
            continue
        
        frame_path = ep_dir / 'frame_0009.png'
        if not frame_path.exists():
            continue
        
        print(f"\n📂 {ep_name[-40:]}")
        
        # 원본 이미지
        img_original = cv2.imread(str(frame_path))
        stats_original = analyze_rgb(img_original)
        
        print(f"\n원본:")
        print(f"  RGB: ({stats_original['r_mean']:.1f}, {stats_original['g_mean']:.1f}, {stats_original['b_mean']:.1f})")
        print(f"  R/G: {stats_original['rg_ratio']:.3f}")
        
        # 방법 1: Global Offset (Red +8)
        img_method1 = method1_global_offset(img_original, offset=8)
        stats_method1 = analyze_rgb(img_method1)
        
        print(f"\nMethod 1 (Red +8):")
        print(f"  RGB: ({stats_method1['r_mean']:.1f}, {stats_method1['g_mean']:.1f}, {stats_method1['b_mean']:.1f})")
        print(f"  R/G: {stats_method1['rg_ratio']:.3f}")
        
        # 방법 2: White Balance + Red boost 1.1x
        img_method2 = method2_white_balance_enhanced(img_original, red_boost=1.1)
        stats_method2 = analyze_rgb(img_method2)
        
        print(f"\nMethod 2 (WB + Red×1.1):")
        print(f"  RGB: ({stats_method2['r_mean']:.1f}, {stats_method2['g_mean']:.1f}, {stats_method2['b_mean']:.1f})")
        print(f"  R/G: {stats_method2['rg_ratio']:.3f}")
        
        # 방법 3: Ratio-based (R/G = 1.05)
        img_method3 = method3_ratio_based_enhanced(img_original, target_rg=1.05)
        stats_method3 = analyze_rgb(img_method3)
        
        print(f"\nMethod 3 (R/G=1.05):")
        print(f"  RGB: ({stats_method3['r_mean']:.1f}, {stats_method3['g_mean']:.1f}, {stats_method3['b_mean']:.1f})")
        print(f"  R/G: {stats_method3['rg_ratio']:.3f}")
        
        # 비교 이미지 생성
        create_comparison(
            img_original, img_method1, img_method2, img_method3,
            stats_original, stats_method1, stats_method2, stats_method3,
            output_dir / f"{ep_name}_comparison_enhanced.png"
        )
        
        all_results.append({
            'episode': ep_name,
            'original': stats_original,
            'method1': stats_method1,
            'method2': stats_method2,
            'method3': stats_method3
        })
    
    # 요약
    print("\n" + "=" * 80)
    print("📊 보정 결과 요약 (Red 강화)")
    print("=" * 80)
    
    for method_name in ['original', 'method1', 'method2', 'method3']:
        rg_ratios = [r[method_name]['rg_ratio'] for r in all_results]
        print(f"\n{method_name.upper()}:")
        print(f"  평균 R/G: {np.mean(rg_ratios):.3f}")
        if method_name == 'original':
            print(f"  → Red 부족 (Cyan/Blue tint)")
        else:
            print(f"  → Red {(np.mean(rg_ratios) - 1.0)*100:+.1f}% vs Green")
    
    print(f"\n💾 비교 이미지 저장: {output_dir}")
    print("\n✅ Red 강화 테스트 완료!")
    print("\n💡 초록 뚜껑이 더 잘 보이는 방법을 확인하세요!")
    
    return all_results


def create_comparison(img_orig, img_m1, img_m2, img_m3, 
                     stats_orig, stats_m1, stats_m2, stats_m3, output_path):
    """4가지 방법 비교 이미지 생성"""
    
    fig = plt.figure(figsize=(20, 5))
    gs = GridSpec(1, 4, figure=fig)
    
    images = [img_orig, img_m1, img_m2, img_m3]
    stats = [stats_orig, stats_m1, stats_m2, stats_m3]
    titles = [
        'Original\n(Cyan tint)', 
        'Method 1\n(Red +8)', 
        'Method 2\n(WB + Red×1.1)', 
        'Method 3\n(R/G=1.05)'
    ]
    
    for i, (img, stat, title) in enumerate(zip(images, stats, titles)):
        ax = fig.add_subplot(gs[0, i])
        
        # BGR to RGB for display
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)
        
        # 통계 표시
        info = f"RGB: ({stat['r_mean']:.0f}, {stat['g_mean']:.0f}, {stat['b_mean']:.0f})\n"
        info += f"R/G: {stat['rg_ratio']:.3f}"
        
        ax.set_title(f"{title}\n{info}", fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    test_sample_frames_enhanced()
