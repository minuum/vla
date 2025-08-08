#!/usr/bin/env python3
"""
8가지 컵 도달 시나리오 데모 및 가이드

액션 데이터 수집이 성공했으므로 이제 실제 시나리오별 수집을 시작할 수 있습니다!
"""

def show_scenario_guide():
    """8가지 시나리오 수집 가이드"""
    
    print("🎯 컵 도달 시나리오 수집 가이드")
    print("=" * 60)
    print()
    
    print("✅ 액션 데이터 수집 성공 확인!")
    print("   최근 episode_20250808_090709.h5에서:")
    print("   - 전진: 1.25, 횡이동: -1.25 등 실제 액션 기록됨")
    print("   - 움직임 비율: 15% (3/20 프레임)")
    print("   - 🎉 이제 진짜 회피 패턴 학습 가능!")
    print()
    
    scenarios = [
        {
            "key": "1", "name": "1박스-세로-왼쪽경로",
            "setup": "📦 박스 1개를 세로로 중앙에 배치",
            "path": "W W W → A A → W W → D D → 컵",
            "description": "전진 후 왼쪽으로 우회하여 컵 도달"
        },
        {
            "key": "2", "name": "1박스-세로-오른쪽경로", 
            "setup": "📦 박스 1개를 세로로 중앙에 배치",
            "path": "W W → D D → W W W → A A → 컵",
            "description": "전진 후 오른쪽으로 우회하여 컵 도달"
        },
        {
            "key": "3", "name": "1박스-가로-왼쪽경로",
            "setup": "📦 박스 1개를 가로로 중앙에 배치", 
            "path": "W → A A A → W W → D D D → 컵",
            "description": "짧게 전진 후 왼쪽으로 크게 우회"
        },
        {
            "key": "4", "name": "1박스-가로-오른쪽경로",
            "setup": "📦 박스 1개를 가로로 중앙에 배치",
            "path": "W W → D → W W → A → 컵", 
            "description": "전진 후 오른쪽으로 간단 우회"
        },
        {
            "key": "5", "name": "2박스-세로-왼쪽경로",
            "setup": "📦📦 박스 2개를 세로로 나란히 배치",
            "path": "W W → A A A → W W → D D D → 컵",
            "description": "2개 박스를 왼쪽으로 크게 우회"
        },
        {
            "key": "6", "name": "2박스-세로-오른쪽경로",
            "setup": "📦📦 박스 2개를 세로로 나란히 배치", 
            "path": "W → D D D → W W W → A A A → 컵",
            "description": "2개 박스를 오른쪽으로 크게 우회"
        },
        {
            "key": "7", "name": "2박스-가로-왼쪽경로",
            "setup": "📦📦 박스 2개를 가로로 일렬 배치",
            "path": "W → A A A A → W W → D D D D → 컵",
            "description": "2개 박스를 왼쪽으로 매우 크게 우회"
        },
        {
            "key": "8", "name": "2박스-가로-오른쪽경로", 
            "setup": "📦📦 박스 2개를 가로로 일렬 배치",
            "path": "W W → D D → W W → A A → 컵",
            "description": "2개 박스를 오른쪽으로 우회"
        }
    ]
    
    for scenario in scenarios:
        print(f"📌 {scenario['key']}키: {scenario['name']}")
        print(f"   🏗️ 환경: {scenario['setup']}")
        print(f"   🎮 경로: {scenario['path']}")
        print(f"   📝 설명: {scenario['description']}")
        print(f"   🎯 목표: 15개 에피소드")
        print()
    
    print("🚀 사용 방법:")
    print("1. 환경 설정: 박스를 해당 시나리오에 맞게 배치")
    print("2. 시나리오 선택: 1-8 키 중 하나 누르기")
    print("3. 경로 실행: WASD로 제시된 경로 따라가기") 
    print("4. 에피소드 종료: M 키 누르기")
    print("5. 진행상황 확인: P 키 누르기")
    print()
    
    print("📊 전체 목표: 8 × 15 = 120개 에피소드")
    print("💾 진행상황 자동 저장: mobile_vla_dataset/scenario_progress.json")
    print()
    
    print("🎯 핵심 팁:")
    print("✅ 각 시나리오별로 실제 회피 동작 포함시키기")
    print("✅ WASD 예시는 참고용, 실제 상황에 맞게 조정")
    print("✅ 프레임 18개 달성하면 특별 표시됨") 
    print("✅ 액션 데이터가 제대로 수집되는지 확인")
    print()
    
    print("🤖 예상 VLA 학습 결과:")
    print("🔹 상황별 최적 경로 자동 선택")
    print("🔹 장애물 개수/배치 자동 인식")
    print("🔹 왼쪽/오른쪽 상황별 적응적 회피")
    print("🔹 컵 도달 성공률 대폭 향상")

def show_wasd_patterns():
    """WASD 패턴 상세 가이드"""
    
    print("\n" + "="*60)
    print("🎮 WASD 패턴 상세 가이드")  
    print("="*60)
    
    patterns = {
        "기본 동작": {
            "W": "전진 (linear_x: +1.25)",
            "A": "왼쪽 이동 (linear_y: +1.25)", 
            "S": "후진 (linear_x: -1.25)",
            "D": "오른쪽 이동 (linear_y: -1.25)",
            "스페이스": "정지 (모든 값: 0.0)"
        },
        "대각선 동작": {
            "Q": "전진+왼쪽 (linear_x: +1.25, linear_y: +1.25)",
            "E": "전진+오른쪽 (linear_x: +1.25, linear_y: -1.25)",
            "Z": "후진+왼쪽 (linear_x: -1.25, linear_y: +1.25)", 
            "C": "후진+오른쪽 (linear_x: -1.25, linear_y: -1.25)"
        },
        "회전 동작": {
            "R": "왼쪽 회전 (angular_z: +1.25)",
            "T": "오른쪽 회전 (angular_z: -1.25)"
        }
    }
    
    for category, actions in patterns.items():
        print(f"\n📋 {category}:")
        for key, description in actions.items():
            print(f"   {key}: {description}")
    
    print(f"\n⏱️ 타이밍:")
    print(f"   - 키 누름 → 0.3초 동작 → 자동 정지")
    print(f"   - 연속 동작하려면 키를 계속 눌러야 함")
    print(f"   - 각 키 입력마다 액션 데이터 수집됨")

if __name__ == "__main__":
    show_scenario_guide()
    show_wasd_patterns()
    
    print("\n🎉 이제 실제 시나리오별 데이터 수집을 시작하세요!")
    print("💡 colcon build && source install/setup.bash 후 data_collector 실행!")
