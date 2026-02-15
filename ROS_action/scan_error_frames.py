#!/usr/bin/env python3
"""
전체 데이터셋 오류 프레임 스캔 도구
확립된 기준(brightness < 104)을 사용하여 모든 에피소드 검증
"""

import cv2
import numpy as np
from pathlib import Path
import json
from typing import Dict, List
from tqdm import tqdm
from datetime import datetime


def analyze_frame_quick(frame_path: Path) -> Dict:
    """빠른 프레임 분석 (오류 탐지 중심)"""
    img = cv2.imread(str(frame_path))
    if img is None:
        return {'error': 'Failed to load'}
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    mean_brightness = float(np.mean(gray))
    dark_ratio = float(np.sum(gray < 50) / gray.size)
    very_dark_ratio = float(np.sum(gray < 30) / gray.size)
    
    # 오류 판정
    is_error = (
        mean_brightness < 104 or
        dark_ratio > 0.20 or
        very_dark_ratio > 0.15
    )
    
    return {
        'frame_name': frame_path.name,
        'mean_brightness': mean_brightness,
        'dark_ratio': dark_ratio,
        'very_dark_ratio': very_dark_ratio,
        'is_error': is_error
    }


def scan_episode(episode_dir: Path) -> Dict:
    """에피소드 내 모든 프레임 스캔"""
    frames = sorted(episode_dir.glob("frame_*.png"))
    
    frame_results = []
    error_frames = []
    
    for frame_path in frames:
        result = analyze_frame_quick(frame_path)
        if 'error' not in result:
            frame_results.append(result)
            if result['is_error']:
                error_frames.append(result)
    
    # 에피소드 레벨 분류
    error_count = len(error_frames)
    if error_count == 0:
        classification = 'clean'
    elif error_count <= 2:
        classification = 'minor'
    elif error_count <= 5:
        classification = 'moderate'
    else:
        classification = 'severe'
    
    return {
        'episode_name': episode_dir.name,
        'total_frames': len(frame_results),
        'error_frame_count': error_count,
        'classification': classification,
        'error_frames': error_frames,
        'frame_results': frame_results
    }


def scan_dataset(extracted_dir: Path, output_dir: Path):
    """전체 데이터셋 스캔"""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🔍 전체 데이터셋 오류 프레임 스캔 시작")
    print(f"   추출 디렉토리: {extracted_dir}")
    print(f"   오류 기준: brightness < 104 OR dark_ratio > 20% OR very_dark_ratio > 15%")
    print("=" * 80)
    
    # 모든 에피소드 디렉토리 찾기
    episode_dirs = sorted([d for d in extracted_dir.iterdir() if d.is_dir()])
    
    all_results = []
    error_episodes = {
        'clean': [],
        'minor': [],
        'moderate': [],
        'severe': []
    }
    
    # 진행 표시와 함께 스캔
    for episode_dir in tqdm(episode_dirs, desc="에피소드 스캔"):
        result = scan_episode(episode_dir)
        all_results.append(result)
        error_episodes[result['classification']].append(result)
    
    # 통계 요약
    print("\n" + "=" * 80)
    print("📊 스캔 결과 요약")
    print("=" * 80)
    print(f"총 에피소드: {len(all_results)}")
    print(f"  ✅ Clean (0 errors): {len(error_episodes['clean'])} ({len(error_episodes['clean'])/len(all_results)*100:.1f}%)")
    print(f"  ⚠️  Minor (1-2 errors): {len(error_episodes['minor'])} ({len(error_episodes['minor'])/len(all_results)*100:.1f}%)")
    print(f"  ⚠️  Moderate (3-5 errors): {len(error_episodes['moderate'])} ({len(error_episodes['moderate'])/len(all_results)*100:.1f}%)")
    print(f"  ❌ Severe (6+ errors): {len(error_episodes['severe'])} ({len(error_episodes['severe'])/len(all_results)*100:.1f}%)")
    
    total_error_frames = sum(r['error_frame_count'] for r in all_results)
    print(f"\n총 프레임: {sum(r['total_frames'] for r in all_results):,}")
    print(f"오류 프레임: {total_error_frames} ({total_error_frames/(sum(r['total_frames'] for r in all_results))*100:.2f}%)")
    
    # 상세 결과 저장
    summary_data = {
        'scan_date': datetime.now().isoformat(),
        'criteria': {
            'brightness_threshold': 104,
            'dark_ratio_threshold': 0.20,
            'very_dark_ratio_threshold': 0.15
        },
        'summary': {
            'total_episodes': len(all_results),
            'clean': len(error_episodes['clean']),
            'minor': len(error_episodes['minor']),
            'moderate': len(error_episodes['moderate']),
            'severe': len(error_episodes['severe']),
            'total_frames': sum(r['total_frames'] for r in all_results),
            'total_error_frames': total_error_frames
        },
        'episodes': all_results
    }
    
    # JSON 저장
    summary_path = output_dir / 'error_scan_complete.json'
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"\n💾 상세 결과 저장: {summary_path}")
    
    # 문제 에피소드만 별도 저장
    if len(error_episodes['minor']) + len(error_episodes['moderate']) + len(error_episodes['severe']) > 0:
        problematic_path = output_dir / 'problematic_episodes.json'
        with open(problematic_path, 'w') as f:
            json.dump({
                'minor': error_episodes['minor'],
                'moderate': error_episodes['moderate'],
                'severe': error_episodes['severe']
            }, f, indent=2)
        
        print(f"💾 문제 에피소드 목록: {problematic_path}")
    
    return summary_data, error_episodes


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='전체 데이터셋 오류 프레임 스캔')
    parser.add_argument(
        '--extracted-dir',
        type=Path,
        default=Path(__file__).parent / 'extracted_images',
        help='추출된 이미지 디렉토리'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path(__file__).parent / 'error_scan_results',
        help='스캔 결과 출력 디렉토리'
    )
    
    args = parser.parse_args()
    
    summary, episodes = scan_dataset(args.extracted_dir, args.output_dir)
    
    print("\n✅ 스캔 완료!")
