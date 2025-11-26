#!/usr/bin/env python3
"""
재수집을 위한 초기화 스크립트
- scenario_progress.json 초기화
- time_period_stats.json 초기화
- core_patterns.json 초기화 (선택적)
- settings.json 설정 확인
"""
import json
from pathlib import Path
from datetime import datetime

def main():
    dataset_dir = Path('/home/soda/vla/ROS_action/mobile_vla_dataset')
    
    print("=" * 80)
    print("🔄 재수집을 위한 초기화")
    print("=" * 80)
    print()
    
    # 1. scenario_progress.json 초기화
    progress_file = dataset_dir / "scenario_progress.json"
    if progress_file.exists():
        print("📋 scenario_progress.json 초기화 중...")
        data = {
            "last_updated": datetime.now().isoformat(),
            "scenario_stats": {
                "1box_left": 0,
                "1box_right": 0,
                "2box_left": 0,
                "2box_right": 0
            },
            "total_completed": 0,
            "total_target": 1000
        }
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print("✅ scenario_progress.json 초기화 완료")
    else:
        print("⚠️  scenario_progress.json이 없습니다. 생성합니다.")
        data = {
            "last_updated": datetime.now().isoformat(),
            "scenario_stats": {
                "1box_left": 0,
                "1box_right": 0,
                "2box_left": 0,
                "2box_right": 0
            },
            "total_completed": 0,
            "total_target": 1000
        }
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print("✅ scenario_progress.json 생성 완료")
    
    print()
    
    # 2. time_period_stats.json 초기화
    time_period_file = dataset_dir / "time_period_stats.json"
    if time_period_file.exists():
        print("📋 time_period_stats.json 초기화 중...")
        data = {
            "last_updated": datetime.now().isoformat(),
            "time_period_stats": {
                "dawn": 0,
                "morning": 0,
                "evening": 0,
                "night": 0
            },
            "total_completed": 0,
            "total_target": 1000
        }
        with open(time_period_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print("✅ time_period_stats.json 초기화 완료")
    else:
        print("⚠️  time_period_stats.json이 없습니다. 생성합니다.")
        data = {
            "last_updated": datetime.now().isoformat(),
            "time_period_stats": {
                "dawn": 0,
                "morning": 0,
                "evening": 0,
                "night": 0
            },
            "total_completed": 0,
            "total_target": 1000
        }
        with open(time_period_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print("✅ time_period_stats.json 생성 완료")
    
    print()
    
    # 3. settings.json 확인 및 설정
    settings_file = dataset_dir / "settings.json"
    if settings_file.exists():
        print("📋 settings.json 확인 중...")
        with open(settings_file, 'r', encoding='utf-8') as f:
            settings = json.load(f)
        
        current_mode = settings.get('guide_mode', 'dataset')
        print(f"   현재 가이드 모드: {current_mode}")
        
        if current_mode != 'dataset':
            print("   ⚠️  데이터셋 모드를 권장합니다.")
            response = input("   데이터셋 모드로 변경하시겠습니까? (y/n): ")
            if response.lower() == 'y':
                settings['guide_mode'] = 'dataset'
                settings['last_updated'] = datetime.now().isoformat()
                with open(settings_file, 'w', encoding='utf-8') as f:
                    json.dump(settings, f, indent=2, ensure_ascii=False)
                print("   ✅ 가이드 모드를 'dataset'로 변경했습니다.")
            else:
                print("   ℹ️  현재 모드를 유지합니다.")
        else:
            print("   ✅ 데이터셋 모드로 설정되어 있습니다.")
    else:
        print("⚠️  settings.json이 없습니다. 생성합니다.")
        settings = {
            "guide_mode": "dataset",
            "last_updated": datetime.now().isoformat()
        }
        with open(settings_file, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=2, ensure_ascii=False)
        print("✅ settings.json 생성 완료 (가이드 모드: dataset)")
    
    print()
    
    # 4. core_patterns.json 초기화 (선택적)
    core_patterns_file = dataset_dir / "core_patterns.json"
    if core_patterns_file.exists():
        print("📋 core_patterns.json 확인 중...")
        with open(core_patterns_file, 'r', encoding='utf-8') as f:
            patterns = json.load(f)
        
        if patterns:
            print(f"   현재 {len(patterns)}개 가이드가 저장되어 있습니다.")
            response = input("   core_patterns.json을 초기화하시겠습니까? (y/n): ")
            if response.lower() == 'y':
                with open(core_patterns_file, 'w', encoding='utf-8') as f:
                    json.dump({}, f, indent=2, ensure_ascii=False)
                print("   ✅ core_patterns.json 초기화 완료")
            else:
                print("   ℹ️  현재 가이드를 유지합니다.")
        else:
            print("   ✅ core_patterns.json이 비어있습니다.")
    else:
        print("⚠️  core_patterns.json이 없습니다. 생성합니다.")
        with open(core_patterns_file, 'w', encoding='utf-8') as f:
            json.dump({}, f, indent=2, ensure_ascii=False)
        print("✅ core_patterns.json 생성 완료")
    
    print()
    print("=" * 80)
    print("✅ 초기화 완료!")
    print("=" * 80)
    print()
    print("💡 다음 단계:")
    print("   1. 바퀴 성능 테스트")
    print("   2. 가이드 패턴 테스트")
    print("   3. 소규모 수집 테스트")
    print("   4. 본격 수집 시작")

if __name__ == "__main__":
    main()

