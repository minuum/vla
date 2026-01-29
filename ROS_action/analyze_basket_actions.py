
import os
import glob
import h5py
import numpy as np
from datetime import datetime
from collections import defaultdict
import re

DATASET_DIR = "/home/soda/vla/ROS_action/mobile_vla_dataset"

def get_action_code(linear_x, angular_z):
    # 액션 값을 간단한 문자로 변환하여 패턴화
    # 임계값 설정 (노이즈 방지)
    threshold = 0.01
    
    code = ""
    
    if linear_x > threshold:
        code += "W" # 전진
    elif linear_x < -threshold:
        code += "S" # 후진
    else:
        code += "_" # 정지 (선속도 기준)
        
    if angular_z > threshold:
        code += "A" # 좌회전
    elif angular_z < -threshold:
        code += "D" # 우회전
    else:
        code += "_" # 회전 없음
        
    # 간소화
    if code == "W_": return "W" # 직진
    if code == "S_": return "S" # 후진
    if code == "__": return "X" # 정지
    if code == "_A": return "A" # 제자리 좌회전
    if code == "_D": return "D" # 제자리 우회전
    if code == "WA": return "Q" # 전진 좌회전
    if code == "WD": return "E" # 전진 우회전
    return code 

def analyze_actions():
    print(f"📂 Analyzing Action Patterns in {DATASET_DIR}...")
    h5_files = glob.glob(os.path.join(DATASET_DIR, "*basket*.h5"))
    
    # 패턴별 파일 정보 저장: pattern -> list of (timestamp, filename)
    patterns = defaultdict(list)
    
    count_processed = 0
    
    for fpath in h5_files:
        filename = os.path.basename(fpath)
        
        # 타임스탬프 추출
        match = re.search(r'episode_(\d{8})_(\d{6})_', filename)
        if not match:
            continue
            
        date_str = match.group(1)
        time_str = match.group(2)
        try:
            timestamp = datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M%S")
        except:
            continue

        try:
            with h5py.File(fpath, 'r') as f:
                # 프레임 수 확인
                num_frames = f.attrs.get('num_frames', 0)
                if 'images' in f:
                    num_frames = f['images'].shape[0]
                
                if num_frames != 18:
                    continue
                
                # 액션 데이터 로드
                if 'actions' not in f:
                    continue
                    
                actions = f['actions'][:] # shape: (18, 2)
                
                # 액션 시퀀스를 문자열 패턴으로 변환
                chk_str = []
                for i in range(len(actions)):
                    lin_x = actions[i][0]
                    ang_z = actions[i][1]
                    chk_str.append(get_action_code(lin_x, ang_z))
                
                # 패턴 문자열 생성 (예: "W-W-W-A-A-...")
                pattern_key = "-".join(chk_str)
                patterns[pattern_key].append((timestamp, filename))
                count_processed += 1
                
        except Exception as e:
            pass

    print(f"📊 Processed {count_processed} valid 18-frame files.")
    print("-" * 60)
    
    # 빈도수 기준 정렬
    sorted_patterns = sorted(patterns.items(), key=lambda item: len(item[1]), reverse=True)
    
    for rank, (pattern, files) in enumerate(sorted_patterns[:5], 1):
        count = len(files)
        files.sort(key=lambda x: x[0]) # 시간순 정렬
        
        first_ts, first_file = files[0]
        last_ts, last_file = files[-1]
        
        # 패턴 요약 표시
        # 너무 기니까 축약해서 보여줌 (예: Wx5, Ax2 ...)
        compact_pattern = pattern.replace("-", "")
        
        print(f"🏆 Rank {rank}: {count}개 수집됨")
        print(f"   🕹️  패턴: {pattern}")
        print(f"   📅 시작: {first_ts} -> {first_file}")
        print(f"   📅 최근: {last_ts}  -> {last_file}")
        print("")

if __name__ == "__main__":
    analyze_actions()
