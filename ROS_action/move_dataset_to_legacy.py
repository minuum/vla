#!/usr/bin/env python3
"""
바퀴 수리 중 카메라 위치 문제로 인해 수집된 데이터를 legacy 폴더로 이동하는 스크립트
사유: 바퀴 수리 중 카메라가 윗포지션을 잡아서 목표물과 장애물이 다 안보이는 상황
"""

import shutil
from pathlib import Path
from datetime import datetime

# 경로 설정
ros_action_dir = Path("/home/soda/vla/ROS_action")
dataset_dir = ros_action_dir / "mobile_vla_dataset"
legacy_dir = ros_action_dir / "mobile_vla_dataset_legacy"

# legacy 디렉토리 생성
legacy_dir.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("📦 데이터셋을 legacy 폴더로 이동")
print("=" * 60)
print(f"📁 소스: {dataset_dir}")
print(f"📁 대상: {legacy_dir}")
print()

# 이동할 항목들
moved_items = []
skipped_items = []
error_items = []

# 모든 파일과 디렉토리 이동
if dataset_dir.exists():
    for item in dataset_dir.iterdir():
        # 설정 파일들은 백업 후 이동
        if item.name in ['settings.json', 'core_patterns.json']:
            # 백업 파일명 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{item.stem}_backup_{timestamp}{item.suffix}"
            backup_path = legacy_dir / backup_name
            
            try:
                # 백업 복사
                shutil.copy2(item, backup_path)
                print(f"💾 백업: {item.name} → {backup_name}")
                
                # 원본 이동
                dest = legacy_dir / item.name
                if dest.exists():
                    print(f"⚠️  건너뜀 (이미 존재): {item.name}")
                    skipped_items.append(item.name)
                else:
                    shutil.move(str(item), str(dest))
                    print(f"✅ 이동: {item.name}")
                    moved_items.append(item.name)
            except Exception as e:
                print(f"❌ 오류 ({item.name}): {e}")
                error_items.append((item.name, str(e)))
        else:
            # 일반 파일/디렉토리 이동
            dest = legacy_dir / item.name
            try:
                if dest.exists():
                    print(f"⚠️  건너뜀 (이미 존재): {item.name}")
                    skipped_items.append(item.name)
                else:
                    if item.is_dir():
                        shutil.move(str(item), str(dest))
                        print(f"📁 이동 (폴더): {item.name}/")
                    else:
                        shutil.move(str(item), str(dest))
                        print(f"📄 이동 (파일): {item.name}")
                    moved_items.append(item.name)
            except Exception as e:
                print(f"❌ 오류 ({item.name}): {e}")
                error_items.append((item.name, str(e)))

print()
print("=" * 60)
print("📊 이동 결과 요약")
print("=" * 60)
print(f"✅ 이동 완료: {len(moved_items)}개")
print(f"⚠️  건너뜀: {len(skipped_items)}개")
print(f"❌ 오류: {len(error_items)}개")

if moved_items:
    print(f"\n✅ 이동된 항목들:")
    for item in sorted(moved_items)[:20]:  # 처음 20개만 표시
        print(f"   - {item}")
    if len(moved_items) > 20:
        print(f"   ... 외 {len(moved_items) - 20}개")

if skipped_items:
    print(f"\n⚠️  건너뛴 항목들:")
    for item in sorted(skipped_items)[:10]:
        print(f"   - {item}")
    if len(skipped_items) > 10:
        print(f"   ... 외 {len(skipped_items) - 10}개")

if error_items:
    print(f"\n❌ 오류 발생 항목들:")
    for item, error in error_items:
        print(f"   - {item}: {error}")

print()
print("=" * 60)
print("✅ 이동 작업 완료!")
print("=" * 60)
print(f"\n💡 참고: 설정 파일들(settings.json, core_patterns.json)은")
print(f"   백업 파일로도 저장되었습니다.")

