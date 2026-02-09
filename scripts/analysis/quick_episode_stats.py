#!/usr/bin/env python3
"""분석: 에피소드 길이 분포"""

import h5py
import glob
import numpy as np

episodes = sorted(glob.glob('ROS_action/basket_dataset/*.h5'))
lengths = []

print(f"분석 중... (총 {len(episodes)} 에피소드)")

for i, ep_file in enumerate(episodes[:50]):  # 샘플링
    try:
        with h5py.File(ep_file, 'r') as f:
            length = len(f['images'])
            lengths.append(length)
    except:
        pass
    if (i+1) % 10 == 0:
        print(f"  {i+1}/50 완료...")

print(f'\n=== 에피소드 길이 통계 (샘플 50개) ===')        
print(f'Min: {min(lengths)}, Max: {max(lengths)}')
print(f'Mean: {np.mean(lengths):.1f}, Median: {np.median(lengths):.1f}')
print(f'\n분포:')
print(f'  <10 frames: {sum(1 for l in lengths if l < 10)}')
print(f'  10-15: {sum(1 for l in lengths if 10 <= l < 15)}')
print(f'  15-20: {sum(1 for l in lengths if 15 <= l < 20)}')
print(f'  20-25: {sum(1 for l in lengths if 20 <= l < 25)}')
print(f'  25+: {sum(1 for l in lengths if l >= 25)}')
print(f'\n샘플: {sorted(lengths)[:20]}')
