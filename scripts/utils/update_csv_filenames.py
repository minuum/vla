#!/usr/bin/env python3
"""
기존 CSV 파일들의 파일명에 시간대 정보를 추가하는 스크립트
H5 파일의 time_period 메타데이터를 읽어서 CSV 파일명을 갱신합니다.
"""
import h5py
import pandas as pd
from pathlib import Path
from datetime import datetime

def classify_time_period_from_hour(hour: int) -> str:
    """시간(시)을 기반으로 시간대 분류"""
    if 0 <= hour < 6:
        return "dawn"
    elif 6 <= hour < 12:
        return "morning"
    elif 12 <= hour < 18:
        return "evening"
    else:  # 18 <= hour < 24
        return "night"

def get_time_period_from_h5(h5_file: Path) -> str:
    """H5 파일에서 time_period 메타데이터를 읽거나 추정"""
    try:
        with h5py.File(h5_file, 'r') as f:
            # 먼저 time_period 메타데이터 확인
            time_period = f.attrs.get('time_period', None)
            if time_period:
                return time_period.decode('utf-8') if isinstance(time_period, bytes) else str(time_period)
            
            # time_period가 없으면 collection_hour나 파일명에서 추정
            collection_hour = f.attrs.get('collection_hour', None)
            if collection_hour is not None:
                return classify_time_period_from_hour(int(collection_hour))
            
            # 파일명에서 타임스탬프 추출 시도
            # episode_YYYYMMDD_HHMMSS_... 형식
            file_stem = h5_file.stem
            parts = file_stem.split('_')
            if len(parts) >= 2:
                try:
                    # YYYYMMDD_HHMMSS 형식 찾기
                    date_str = parts[1]  # YYYYMMDD
                    time_str = parts[2] if len(parts) > 2 else None  # HHMMSS
                    
                    if time_str and len(time_str) >= 2:
                        hour = int(time_str[:2])
                        return classify_time_period_from_hour(hour)
                except (ValueError, IndexError):
                    pass
            
    except Exception as e:
        print(f"⚠️ H5 파일 읽기 실패 {h5_file.name}: {e}")
    
    return None

def update_csv_filename(csv_file: Path, dataset_dir: Path) -> bool:
    """CSV 파일명을 시간대 정보가 포함된 이름으로 갱신"""
    # CSV 파일명에서 H5 파일명 추출
    # episode_20251106_151851_1box_hori_left_core_medium_data.csv
    # -> episode_20251106_151851_1box_hori_left_core_medium.h5
    
    csv_stem = csv_file.stem  # episode_..._medium_data
    if csv_stem.endswith('_data'):
        h5_stem = csv_stem[:-5]  # _data 제거
    else:
        h5_stem = csv_stem
    
    h5_file = dataset_dir / f"{h5_stem}.h5"
    
    if not h5_file.exists():
        print(f"⚠️ 해당 H5 파일을 찾을 수 없습니다: {h5_file.name}")
        return False
    
    # H5 파일에서 time_period 읽기
    time_period = get_time_period_from_h5(h5_file)
    
    if not time_period:
        print(f"⚠️ 시간대 정보를 추출할 수 없습니다: {h5_file.name}")
        return False
    
    # 새로운 파일명 생성
    # medium 뒤에 시간대 추가
    if 'medium' in h5_stem:
        new_stem = h5_stem.replace('medium', f'medium_{time_period}')
    else:
        new_stem = f"{h5_stem}_{time_period}"
    
    new_csv_file = dataset_dir / f"{new_stem}_data.csv"
    
    # 이미 올바른 이름이면 스킵
    if csv_file.name == new_csv_file.name:
        print(f"✓ 이미 올바른 이름입니다: {csv_file.name}")
        return True
    
    # CSV 파일 읽기
    try:
        df = pd.read_csv(csv_file)
        
        # 새 파일명으로 저장
        df.to_csv(new_csv_file, index=False)
        
        # 기존 파일 삭제
        csv_file.unlink()
        
        print(f"✅ 갱신 완료: {csv_file.name} -> {new_csv_file.name}")
        return True
        
    except Exception as e:
        print(f"❌ CSV 파일 갱신 실패 {csv_file.name}: {e}")
        return False

def main():
    """메인 함수"""
    dataset_dir = Path("/home/soda/vla/ROS_action/mobile_vla_dataset")
    
    if not dataset_dir.exists():
        print(f"❌ 데이터셋 디렉토리를 찾을 수 없습니다: {dataset_dir}")
        return
    
    # 모든 CSV 파일 찾기
    csv_files = list(dataset_dir.glob("*_data.csv"))
    
    if not csv_files:
        print("📁 갱신할 CSV 파일이 없습니다.")
        return
    
    print(f"📁 총 {len(csv_files)}개의 CSV 파일을 확인합니다...\n")
    
    updated_count = 0
    skipped_count = 0
    failed_count = 0
    
    for csv_file in csv_files:
        # 파일명에 이미 시간대 정보가 있는지 확인
        # medium_dawn, medium_morning, medium_evening, medium_night 패턴 확인
        if any(f'medium_{period}' in csv_file.stem for period in ['dawn', 'morning', 'evening', 'night']):
            print(f"⏭️  이미 시간대 정보가 포함되어 있습니다: {csv_file.name}")
            skipped_count += 1
            continue
        
        if update_csv_filename(csv_file, dataset_dir):
            updated_count += 1
        else:
            failed_count += 1
    
    print("\n" + "="*50)
    print(f"📊 갱신 완료:")
    print(f"   ✅ 갱신됨: {updated_count}개")
    print(f"   ⏭️  스킵됨: {skipped_count}개")
    print(f"   ❌ 실패: {failed_count}개")
    print("="*50)

if __name__ == "__main__":
    main()

