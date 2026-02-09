#!/usr/bin/env python3
"""
상세 오류 분석 스크립트
- 구간별(초반/중반/후반) 성능 분석
- 프레임별 오류 패턴 분석
- 실패 원인 분류 (Stop confusion, Magnitude error, Direction error 등)
"""

import os
import sys
import json
import h5py
import requests
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict, Counter

API_BASE = "http://localhost:8000"
# Try multiple possible API keys
POSSIBLE_KEYS = [
    "vla-mobile-fixed-key-20260205",  # Actual key from process environment
    "nozI8wvqUV1i_4iESNwn-2D5x-Mf-JUJjvXKwOy0C00",  # Current server key
    os.getenv("VLA_API_KEY"),
    "0Yf5D1z0AfjMY5aqtBVBvskHVeNjIo5uKYwH7bGfkKE",
    "35bnTxV_Wa7zD8WhmWb-lUMCjt0EAqT8gV6H8lpt7LA",
    "vla_mobile_robot_2025",
    "mobile-vla-key-2024"
]
API_KEY = next((k for k in POSSIBLE_KEYS if k), "mobile-vla-key-2024")
HEADERS = {"X-API-Key": API_KEY}

# 테스트 설정
DATASET_DIR = "/home/billy/25-1kp/vla/ROS_action/basket_dataset_v2/test"
NUM_EPISODES = 20  # 더 많은 에피소드로 테스트

# 분석 임계값
TOLERANCE = 0.01  # Perfect Match 허용 오차
DIRECTION_THRESHOLD = 0.0  # 방향 일치 판단 기준

class ErrorClassifier:
    """오류 유형 분류기"""
    
    @staticmethod
    def classify_error(gt, pred):
        """
        오류 유형 분류
        
        Returns:
            error_type: str
                - 'perfect': 완벽 일치
                - 'stop_confusion': 정지 구간 혼동
                - 'magnitude_under': 크기 과소 예측
                - 'magnitude_over': 크기 과다 예측
                - 'direction_flip': 방향 반대
                - 'minor_deviation': 경미한 편차
        """
        # Ensure arrays are 1D with 2 elements
        gt = np.array(gt).flatten()[:2]
        pred = np.array(pred).flatten()[:2]
        gt_x, gt_y = gt
        pred_x, pred_y = pred
        
        # Perfect match
        if np.allclose(gt, pred, atol=TOLERANCE):
            return 'perfect'
        
        # Stop confusion (GT는 정지인데 움직임 예측, 또는 그 반대)
        gt_stopped = np.allclose(gt, [0, 0], atol=0.05)
        pred_stopped = np.allclose(pred, [0, 0], atol=0.05)
        
        if gt_stopped and not pred_stopped:
            return 'stop_confusion_false_move'
        if not gt_stopped and pred_stopped:
            return 'stop_confusion_false_stop'
        
        # Direction flip (부호 반대)
        if (gt_x * pred_x < 0) or (gt_y * pred_y < 0):
            return 'direction_flip'
        
        # Magnitude errors
        mag_gt = np.linalg.norm(gt)
        mag_pred = np.linalg.norm(pred)
        
        if mag_pred < mag_gt * 0.7:  # 30% 이상 작음
            return 'magnitude_under'
        if mag_pred > mag_gt * 1.3:  # 30% 이상 큼
            return 'magnitude_over'
        
        # Minor deviation
        return 'minor_deviation'

def analyze_episode(h5_path, api_base=API_BASE):
    """에피소드 상세 분석"""
    
    with h5py.File(h5_path, 'r') as f:
        images = f['images'][:]
        actions = f['actions'][:]
        # Language instruction might not exist in this format
        try:
            language = f['language'][()].decode('utf-8')
        except:
            language = "Navigate to the basket"  # Default instruction
    
    num_frames = len(images)
    episode_name = h5_path.stem
    
    # 구간 정의
    phase_ranges = {
        'initial': (0, min(5, num_frames)),
        'middle': (5, max(5, num_frames - 5)),
        'final': (max(5, num_frames - 5), num_frames)
    }
    
    results = {
        'episode_name': episode_name,
        'total_frames': num_frames,
        'language': language,
        'phases': {},
        'frame_details': [],
        'error_distribution': Counter()
    }
    
    # 프레임별 분석
    for i in tqdm(range(num_frames), desc=f"  {episode_name[:30]}", leave=False):
        # API 호출
        import base64
        from PIL import Image
        import io
        
        img_pil = Image.fromarray(images[i])
        buf = io.BytesIO()
        img_pil.save(buf, format='PNG')
        img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        # Try API call with multiple keys
        pred_action = None
        last_error = None
        
        for key in POSSIBLE_KEYS:
            if not key:
                continue
            try:
                response = requests.post(
                    f"{api_base}/predict",
                    headers={"X-API-Key": key},
                    json={
                        "image": img_b64,
                        "instruction": language
                    },
                    timeout=5
                )
                response.raise_for_status()
                pred_action = np.array(response.json()['action'])
                break  # Success!
            except Exception as e:
                last_error = e
                continue
        
        if pred_action is None:
            if i == 0:  # Only print first frame error
                print(f"  ⚠️  API Error (trying all keys failed): {last_error}")
            pred_action = np.array([0.0, 0.0])
        
        gt_action = actions[i][:2]  # Only use first 2 dimensions (linear_x, linear_y)
        
        # 오류 분류
        error_type = ErrorClassifier.classify_error(gt_action, pred_action)
        results['error_distribution'][error_type] += 1
        
        # Perfect Match & Direction Agreement
        perfect = np.allclose(gt_action, pred_action, atol=TOLERANCE)
        dir_match = (np.sign(gt_action[0]) == np.sign(pred_action[0])) and \
                    (np.sign(gt_action[1]) == np.sign(pred_action[1]))
        
        # 프레임별 상세 정보
        frame_info = {
            'frame_idx': i,
            'gt_action': gt_action.tolist(),
            'pred_action': pred_action.tolist(),
            'error': np.linalg.norm(gt_action - pred_action),
            'perfect_match': perfect,
            'direction_match': dir_match,
            'error_type': error_type
        }
        results['frame_details'].append(frame_info)
    
    # 구간별 통계
    for phase, (start, end) in phase_ranges.items():
        phase_frames = results['frame_details'][start:end]
        if not phase_frames:
            continue
            
        pm_count = sum(1 for f in phase_frames if f['perfect_match'])
        dm_count = sum(1 for f in phase_frames if f['direction_match'])
        
        results['phases'][phase] = {
            'range': (start, end),
            'perfect_match': pm_count / len(phase_frames) * 100,
            'direction_agreement': dm_count / len(phase_frames) * 100,
            'avg_error': np.mean([f['error'] for f in phase_frames])
        }
    
    # 전체 통계
    pm_total = sum(1 for f in results['frame_details'] if f['perfect_match'])
    dm_total = sum(1 for f in results['frame_details'] if f['direction_match'])
    
    results['overall'] = {
        'perfect_match': pm_total / num_frames * 100,
        'direction_agreement': dm_total / num_frames * 100,
        'avg_error': np.mean([f['error'] for f in results['frame_details']])
    }
    
    return results

def main():
    print("🔍 상세 오류 분석 시작")
    print(f"📂 Dataset: {DATASET_DIR}")
    print(f"📊 Target: {NUM_EPISODES} episodes")
    print(f"🔧 Tolerance: {TOLERANCE}, Direction Threshold: {DIRECTION_THRESHOLD}")
    print()
    
    # Dataset 로드
    dataset_path = Path(DATASET_DIR)
    h5_files = sorted(list(dataset_path.glob("*.h5")))[:NUM_EPISODES]
    
    print(f"Found {len(h5_files)} episodes")
    print()
    
    # 에피소드별 분석
    all_results = []
    aggregated_errors = Counter()
    
    for h5_path in h5_files:
        print(f"Processing: {h5_path.name}")
        result = analyze_episode(h5_path)
        all_results.append(result)
        
        # 오류 분포 집계
        aggregated_errors.update(result['error_distribution'])
        
        # 요약 출력
        print(f"  Overall: PM={result['overall']['perfect_match']:.1f}%, DA={result['overall']['direction_agreement']:.1f}%")
        print(f"  Phases:")
        for phase, stats in result['phases'].items():
            print(f"    {phase:8s}: PM={stats['perfect_match']:.1f}%, DA={stats['direction_agreement']:.1f}%")
        print()
    
    # 전역 통계
    total_frames = sum(r['total_frames'] for r in all_results)
    global_pm = np.mean([r['overall']['perfect_match'] for r in all_results])
    global_da = np.mean([r['overall']['direction_agreement'] for r in all_results])
    
    print("="*80)
    print("📊 전역 통계 (Global Statistics)")
    print("="*80)
    print(f"Total Episodes: {len(all_results)}")
    print(f"Total Frames: {total_frames}")
    print(f"Global Perfect Match: {global_pm:.2f}%")
    print(f"Global Direction Agreement: {global_da:.2f}%")
    print()
    
    # 오류 분포
    print("📉 오류 유형 분포 (Error Distribution)")
    print("-"*80)
    total_errors = sum(aggregated_errors.values())
    for error_type, count in aggregated_errors.most_common():
        percentage = count / total_errors * 100
        print(f"{error_type:30s}: {count:4d} ({percentage:5.1f}%)")
    print()
    
    # 구간별 평균
    print("📍 구간별 평균 성능 (Phase-wise Performance)")
    print("-"*80)
    for phase in ['initial', 'middle', 'final']:
        phase_pms = [r['phases'][phase]['perfect_match'] for r in all_results if phase in r['phases']]
        phase_das = [r['phases'][phase]['direction_agreement'] for r in all_results if phase in r['phases']]
        
        if phase_pms:
            print(f"{phase:10s}: PM={np.mean(phase_pms):.2f}%, DA={np.mean(phase_das):.2f}%")
    print()
    
    # 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"logs/detailed_error_analysis_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump({
            'test_config': {
                'num_episodes': len(all_results),
                'total_frames': total_frames,
                'tolerance': TOLERANCE
            },
            'global_stats': {
                'perfect_match': global_pm,
                'direction_agreement': global_da
            },
            'error_distribution': dict(aggregated_errors),
            'episode_results': all_results
        }, f, indent=2)
    
    print(f"💾 Results saved: {output_file}")

if __name__ == "__main__":
    main()
