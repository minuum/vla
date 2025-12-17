#!/usr/bin/env python3
"""
오류 이미지 분석 도구
frame_0006, frame_0007을 정상 프레임과 비교하여 명확한 오류 기준 수립
"""

import cv2
import numpy as np
from pathlib import Path
import json


def analyze_frame(frame_path: Path):
    """개별 프레임 상세 분석"""
    img = cv2.imread(str(frame_path))
    if img is None:
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 상세 통계
    stats = {
        'path': str(frame_path),
        'frame_name': frame_path.name,
        'shape': img.shape,
        
        # 밝기 통계
        'mean_brightness': float(np.mean(gray)),
        'std_brightness': float(np.std(gray)),
        'min_brightness': int(np.min(gray)),
        'max_brightness': int(np.max(gray)),
        
        # 히스토그램 분석
        'histogram_peak': int(np.argmax(np.histogram(gray, bins=256)[0])),
        
        # 어두운 픽셀 비율
        'dark_pixel_ratio': float(np.sum(gray < 50) / gray.size),
        'very_dark_ratio': float(np.sum(gray < 30) / gray.size),
        
        # 밝은 픽셀 비율
        'bright_pixel_ratio': float(np.sum(gray > 200) / gray.size),
        
        # 엔트로피
        'entropy': calculate_entropy(gray)
    }
    
    return stats


def calculate_entropy(gray_img):
    """이미지 엔트로피 계산"""
    hist, _ = np.histogram(gray_img, bins=256, range=(0, 256))
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist))
    return float(entropy)


def compare_frames(episode_dir: Path):
    """에피소드 내 모든 프레임 비교"""
    
    print(f"🔍 에피소드 분석: {episode_dir.name}\n")
    print("=" * 80)
    
    frames = sorted(episode_dir.glob("frame_*.png"))
    
    all_stats = []
    for frame_path in frames:
        stats = analyze_frame(frame_path)
        if stats:
            all_stats.append(stats)
            
            # 오류 프레임 표시
            is_error = frame_path.name in ['frame_0006.png', 'frame_0007.png']
            marker = "⚠️ ERROR" if is_error else "✅"
            
            print(f"{marker} {stats['frame_name']}")
            print(f"   평균 밝기: {stats['mean_brightness']:.1f}")
            print(f"   표준편차: {stats['std_brightness']:.1f}")
            print(f"   어두운 픽셀 (< 50): {stats['dark_pixel_ratio']*100:.1f}%")
            print(f"   매우 어두운 픽셀 (< 30): {stats['very_dark_ratio']*100:.1f}%")
            print(f"   엔트로피: {stats['entropy']:.2f}")
            print()
    
    # 통계 요약
    print("=" * 80)
    print("📊 통계 비교\n")
    
    # 프레임 그룹 분리
    error_frames = [s for s in all_stats if s['frame_name'] in ['frame_0006.png', 'frame_0007.png']]
    normal_frames = [s for s in all_stats if s['frame_name'] not in ['frame_0006.png', 'frame_0007.png']]
    
    if error_frames:
        print("오류 프레임 (frame_0006, frame_0007):")
        print(f"  평균 밝기: {np.mean([s['mean_brightness'] for s in error_frames]):.1f}")
        print(f"  범위: {np.min([s['mean_brightness'] for s in error_frames]):.1f} ~ {np.max([s['mean_brightness'] for s in error_frames]):.1f}")
        print(f"  어두운 픽셀 비율: {np.mean([s['dark_pixel_ratio'] for s in error_frames])*100:.1f}%")
        print()
    
    if normal_frames:
        print("정상 프레임 (나머지):")
        print(f"  평균 밝기: {np.mean([s['mean_brightness'] for s in normal_frames]):.1f}")
        print(f"  표준편차: {np.std([s['mean_brightness'] for s in normal_frames]):.1f}")
        print(f"  범위: {np.min([s['mean_brightness'] for s in normal_frames]):.1f} ~ {np.max([s['mean_brightness'] for s in normal_frames]):.1f}")
        print(f"  어두운 픽셀 비율: {np.mean([s['dark_pixel_ratio'] for s in normal_frames])*100:.1f}%")
        print()
    
    # 권장 임계값
    if error_frames and normal_frames:
        error_brightness = np.max([s['mean_brightness'] for s in error_frames])
        normal_brightness = np.min([s['mean_brightness'] for s in normal_frames])
        
        threshold = (error_brightness + normal_brightness) / 2
        
        print("=" * 80)
        print("🎯 권장 오류 탐지 기준\n")
        print(f"평균 밝기 임계값: < {threshold:.1f}")
        print(f"  오류 프레임 최대: {error_brightness:.1f}")
        print(f"  정상 프레임 최소: {normal_brightness:.1f}")
        print()
    
    # 결과 저장
    output_file = episode_dir.parent / 'error_analysis_results.json'
    with open(output_file, 'w') as f:
        json.dump({
            'episode': episode_dir.name,
            'total_frames': len(all_stats),
            'error_frames': error_frames,
            'normal_frames_stats': {
                'count': len(normal_frames),
                'mean_brightness': float(np.mean([s['mean_brightness'] for s in normal_frames])) if normal_frames else 0,
                'std_brightness': float(np.std([s['mean_brightness'] for s in normal_frames])) if normal_frames else 0,
            },
            'all_stats': all_stats
        }, f, indent=2)
    
    print(f"💾 상세 결과 저장: {output_file}")
    
    return all_stats, error_frames, normal_frames


if __name__ == '__main__':
    episode_dir = Path(__file__).parent / 'mobile_vla_dataset' / 'episode_20251204_000842_1box_hori_right_core_medium'
    
    all_stats, error_frames, normal_frames = compare_frames(episode_dir)
