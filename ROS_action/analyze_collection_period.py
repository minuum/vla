#!/usr/bin/env python3
"""
84개 데이터셋 수집 기간 분석 스크립트
"""

import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# 데이터셋 디렉토리
dataset_dir = Path("/home/soda/vla/ROS_action/mobile_vla_dataset")

# H5 파일 목록 가져오기
h5_files = sorted(dataset_dir.glob("episode_20251127_*.h5"))

if not h5_files:
    print("❌ H5 파일을 찾을 수 없습니다.")
    exit(1)

# 타임스탬프 추출
timestamps = []
for h5_file in h5_files:
    # 파일명에서 타임스탬프 추출: episode_20251127_HHMMSS_...
    match = re.search(r'episode_20251127_(\d{6})_', h5_file.name)
    if match:
        timestamp_str = match.group(1)
        # HHMMSS를 datetime으로 변환
        hour = int(timestamp_str[0:2])
        minute = int(timestamp_str[2:4])
        second = int(timestamp_str[4:6])
        timestamps.append((hour, minute, second, h5_file.name))

# 시간순 정렬
timestamps.sort()

# 첫 번째와 마지막 파일 확인
first_file = timestamps[0]
last_file = timestamps[-1]

print("=" * 60)
print("📊 데이터셋 수집 기간 분석")
print("=" * 60)
print(f"\n📁 총 파일 수: {len(h5_files)}개")
print(f"\n⏰ 수집 시작: {first_file[0]:02d}:{first_file[1]:02d}:{first_file[2]:02d}")
print(f"   파일: {first_file[3]}")
print(f"\n⏰ 수집 종료: {last_file[0]:02d}:{last_file[1]:02d}:{last_file[2]:02d}")
print(f"   파일: {last_file[3]}")

# 시간대별 그룹화 (연속된 수집 구간 찾기)
time_groups = []
current_group = [timestamps[0]]

for i in range(1, len(timestamps)):
    prev_time = timestamps[i-1]
    curr_time = timestamps[i]
    
    # 이전 시간과 현재 시간의 차이 계산 (초 단위)
    prev_seconds = prev_time[0] * 3600 + prev_time[1] * 60 + prev_time[2]
    curr_seconds = curr_time[0] * 3600 + curr_time[1] * 60 + curr_time[2]
    time_diff = curr_seconds - prev_seconds
    
    # 5분(300초) 이상 간격이 있으면 새로운 그룹으로 간주
    if time_diff > 300:
        time_groups.append(current_group)
        current_group = [curr_time]
    else:
        current_group.append(curr_time)

# 마지막 그룹 추가
if current_group:
    time_groups.append(current_group)

print(f"\n📋 수집 구간: {len(time_groups)}개")
print("=" * 60)

for idx, group in enumerate(time_groups, 1):
    start = group[0]
    end = group[-1]
    
    # 시작 시간과 종료 시간 포맷
    start_str = f"{start[0]:02d}:{start[1]:02d}:{start[2]:02d}"
    end_str = f"{end[0]:02d}:{end[1]:02d}:{end[2]:02d}"
    
    # 소요 시간 계산
    start_seconds = start[0] * 3600 + start[1] * 60 + start[2]
    end_seconds = end[0] * 3600 + end[1] * 60 + end[2]
    duration_seconds = end_seconds - start_seconds
    duration_minutes = duration_seconds / 60
    
    print(f"\n구간 {idx}: {start_str} ~ {end_str}")
    print(f"   파일 수: {len(group)}개")
    print(f"   소요 시간: {duration_minutes:.1f}분 ({duration_seconds}초)")

print("\n" + "=" * 60)
print("✅ 분석 완료")
print("=" * 60)

