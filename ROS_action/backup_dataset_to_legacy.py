#!/usr/bin/env python3
"""
데이터셋 백업 스크립트
mobile_vla_dataset의 모든 파일을 mobile_vla_dataset_legacy로 이동
"""
import shutil
from pathlib import Path
from datetime import datetime

def main():
    dataset_dir = Path('/home/soda/vla/ROS_action/mobile_vla_dataset')
    legacy_dir = Path('/home/soda/vla/ROS_action/mobile_vla_dataset_legacy')
    
    # Legacy 디렉토리 생성
    legacy_dir.mkdir(parents=True, exist_ok=True)
    
    # 백업할 파일 목록
    h5_files = list(dataset_dir.glob('*.h5'))
    json_files = list(dataset_dir.glob('*.json'))
    
    print("=" * 80)
    print("📦 데이터셋 백업 시작")
    print("=" * 80)
    print(f"📁 소스: {dataset_dir}")
    print(f"📁 대상: {legacy_dir}")
    print(f"📊 H5 파일: {len(h5_files)}개")
    print(f"📊 JSON 파일: {len(json_files)}개")
    print()
    
    # H5 파일 이동
    moved_h5 = 0
    skipped_h5 = 0
    for h5_file in h5_files:
        dest = legacy_dir / h5_file.name
        if dest.exists():
            print(f"⚠️  건너뜀 (이미 존재): {h5_file.name}")
            skipped_h5 += 1
        else:
            try:
                shutil.move(str(h5_file), str(dest))
                moved_h5 += 1
                if moved_h5 % 10 == 0:
                    print(f"✅ 이동 완료: {moved_h5}/{len(h5_files)}")
            except Exception as e:
                print(f"❌ 이동 실패: {h5_file.name} - {e}")
    
    # JSON 파일 백업 (이름에 타임스탬프 추가)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    moved_json = 0
    for json_file in json_files:
        # 백업 파일명: 원본이름_backup_타임스탬프.json
        backup_name = f"{json_file.stem}_backup_{timestamp}.json"
        dest = legacy_dir / backup_name
        try:
            shutil.copy2(str(json_file), str(dest))
            moved_json += 1
            print(f"✅ JSON 백업: {json_file.name} → {backup_name}")
        except Exception as e:
            print(f"❌ JSON 백업 실패: {json_file.name} - {e}")
    
    print()
    print("=" * 80)
    print("📊 백업 완료 요약")
    print("=" * 80)
    print(f"✅ H5 파일 이동: {moved_h5}개")
    if skipped_h5 > 0:
        print(f"⚠️  H5 파일 건너뜀: {skipped_h5}개")
    print(f"✅ JSON 파일 백업: {moved_json}개")
    print(f"📁 Legacy 디렉토리: {legacy_dir}")
    print()
    print("💡 다음 단계:")
    print("   1. 가이드 재설정 (core_patterns.json 초기화 또는 수정)")
    print("   2. settings.json에서 guide_mode 확인")
    print("   3. 재수집 시작")

if __name__ == "__main__":
    main()

