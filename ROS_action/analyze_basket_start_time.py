
import os
import glob
import h5py
from datetime import datetime
from collections import defaultdict
import re

DATASET_DIR = "/home/soda/vla/ROS_action/mobile_vla_dataset"

def extract_scenario_and_timestamp(filename):
    # 파일명 예시: episode_20251203_123456_basket_1box_hori_left_core_medium.h5
    
    # 1. 시나리오 매핑
    mapping = {
        "basket_1box_vert_left": "basket_1box_left", "basket_1box_hori_left": "basket_1box_left",
        "basket_1box_vert_right": "basket_1box_right", "basket_1box_hori_right": "basket_1box_right",
        "basket_2box_vert_left": "basket_2box_left", "basket_2box_hori_left": "basket_2box_left",
        "basket_2box_vert_right": "basket_2box_right", "basket_2box_hori_right": "basket_2box_right"
    }
    
    scenario = None
    
    # 매핑 테이블 먼저 검사
    for k, v in mapping.items():
        if k in filename:
            scenario = v
            break
            
    # 매핑되지 않았지만 basket으로 시작하는 경우 (구형 파일 등)
    if not scenario and "basket_" in filename:
        # basket_1box_left_core... -> basket_1box_left 추출 시도
        parts = filename.split('_')
        # episode, date, time, [basket, 1box, left/right] ...
        # 보통 basket이 3번째 인덱스 이후에 나옴
        try:
            basket_idx = parts.index("basket")
            # basket + 1box + left/right 조합 확인
            if len(parts) > basket_idx + 2:
                candidate = f"{parts[basket_idx]}_{parts[basket_idx+1]}_{parts[basket_idx+2]}"
                if candidate in ["basket_1box_left", "basket_1box_right", "basket_2box_left", "basket_2box_right"]:
                    scenario = candidate
        except ValueError:
            pass

    if not scenario:
        return None, None

    # 2. 타임스탬프 추출
    # episode_YYYYMMDD_HHMMSS_...
    match = re.search(r'episode_(\d{8})_(\d{6})_', filename)
    if match:
        date_str = match.group(1)
        time_str = match.group(2)
        try:
            timestamp = datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M%S")
            return scenario, timestamp
        except ValueError:
            return None, None
            
    return None, None

def analyze():
    print(f"📂 Analyzing dataset in {DATASET_DIR}...")
    h5_files = glob.glob(os.path.join(DATASET_DIR, "*.h5"))
    
    # 시나리오별 (timestamp, filename) 리스트 저장
    scenario_files = defaultdict(list)
    
    count_total = 0
    count_basket = 0
    count_18frames = 0
    
    for fpath in h5_files:
        count_total += 1
        fname = os.path.basename(fpath)
        
        scenario, timestamp = extract_scenario_and_timestamp(fname)
        if not scenario:
            continue
            
        count_basket += 1
        
        try:
            with h5py.File(fpath, 'r') as f:
                num_frames = f.attrs.get('num_frames', 0)
                if 'images' in f:
                    num_frames = f['images'].shape[0]
                
                if num_frames == 18:
                    count_18frames += 1
                    scenario_files[scenario].append((timestamp, fname))
                    
        except Exception as e:
            # print(f"⚠️ Error reading {fname}: {e}")
            pass

    print(f"📊 Total files: {count_total}")
    print(f"🧺 Basket-related files: {count_basket}")
    print(f"🎥 18-frame Basket files: {count_18frames}")
    print("-" * 60)
    
    for scenario in sorted(scenario_files.keys()):
        files = scenario_files[scenario]
        if not files:
            continue
            
        # 시간순 정렬
        files.sort(key=lambda x: x[0])
        
        first_ts, first_file = files[0]
        last_ts, last_file = files[-1]
        count = len(files)
        
        print(f"✅ {scenario} (총 {count}개)")
        print(f"   📅 시작 시점: {first_ts} -> {first_file}")
        print(f"   📅 최근 시점: {last_ts}  -> {last_file}")
        print("")

if __name__ == "__main__":
    analyze()
