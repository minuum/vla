#!/usr/bin/env python3
"""
컵 도달 태스크를 위한 8가지 시나리오 구성 전략

목표: 컵 앞에 도착하기
장애물: 작은 박스 2개
변수: 장애물 개수 (1개/2개) × 배치 (세로/가로) × 경로 (왼쪽/오른쪽) = 8가지
"""

import json
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class Scenario:
    id: str
    name: str
    description: str
    obstacle_count: int  # 1 or 2
    obstacle_layout: str  # "vertical" or "horizontal"
    preferred_path: str  # "left" or "right"
    target_episodes: int
    example_path: str

class CupReachingStrategy:
    def __init__(self):
        self.scenarios = self.define_scenarios()
        
    def define_scenarios(self) -> List[Scenario]:
        """8가지 시나리오 정의"""
        scenarios = [
            # 1개 장애물 시나리오
            Scenario(
                id="1box_vert_left",
                name="1박스-세로-왼쪽경로",
                description="박스 1개를 세로로 배치, 왼쪽으로 우회하여 컵 도달",
                obstacle_count=1,
                obstacle_layout="vertical", 
                preferred_path="left",
                target_episodes=15,
                example_path="""
시작→ □ → ⬛ → □ ← 컵
     ↓   ↗
     □ ← □
                """
            ),
            Scenario(
                id="1box_vert_right", 
                name="1박스-세로-오른쪽경로",
                description="박스 1개를 세로로 배치, 오른쪽으로 우회하여 컵 도달",
                obstacle_count=1,
                obstacle_layout="vertical",
                preferred_path="right", 
                target_episodes=15,
                example_path="""
시작→ □ → □ → ⬛ → 컵
     ↘   ↗
     □ → □
                """
            ),
            Scenario(
                id="1box_hori_left",
                name="1박스-가로-왼쪽경로", 
                description="박스 1개를 가로로 배치, 왼쪽으로 우회하여 컵 도달",
                obstacle_count=1,
                obstacle_layout="horizontal",
                preferred_path="left",
                target_episodes=15,
                example_path="""
시작→ □ ← 컵
     ↓   ↑
     □ → ⬛ → □
                """
            ),
            Scenario(
                id="1box_hori_right",
                name="1박스-가로-오른쪽경로",
                description="박스 1개를 가로로 배치, 오른쪽으로 우회하여 컵 도달", 
                obstacle_count=1,
                obstacle_layout="horizontal",
                preferred_path="right",
                target_episodes=15,
                example_path="""
시작→ □ → 컵
     ↓   ↑
     ⬛ → □
                """
            ),
            # 2개 장애물 시나리오  
            Scenario(
                id="2box_vert_left",
                name="2박스-세로-왼쪽경로",
                description="박스 2개를 세로로 배치, 왼쪽으로 우회하여 컵 도달",
                obstacle_count=2,
                obstacle_layout="vertical",
                preferred_path="left", 
                target_episodes=15,
                example_path="""
시작→ □ → ⬛ → ⬛ ← 컵
     ↓   ↗
     □ ← □ ← □
                """
            ),
            Scenario(
                id="2box_vert_right",
                name="2박스-세로-오른쪽경로", 
                description="박스 2개를 세로로 배치, 오른쪽으로 우회하여 컵 도달",
                obstacle_count=2,
                obstacle_layout="vertical",
                preferred_path="right",
                target_episodes=15,
                example_path="""
시작→ □ → □ → ⬛ → ⬛ → 컵
     ↘   ↗
     □ → □ → □
                """
            ),
            Scenario(
                id="2box_hori_left",
                name="2박스-가로-왼쪽경로",
                description="박스 2개를 가로로 배치, 왼쪽으로 우회하여 컵 도달", 
                obstacle_count=2,
                obstacle_layout="horizontal",
                preferred_path="left",
                target_episodes=15,
                example_path="""
시작→ □ ← 컵
     ↓   ↑
     □ → ⬛ ⬛ → □
                """
            ),
            Scenario(
                id="2box_hori_right", 
                name="2박스-가로-오른쪽경로",
                description="박스 2개를 가로로 배치, 오른쪽으로 우회하여 컵 도달",
                obstacle_count=2,
                obstacle_layout="horizontal", 
                preferred_path="right",
                target_episodes=15,
                example_path="""
시작→ □ → 컵
     ↓   ↑  
     ⬛ ⬛ → □
                """
            )
        ]
        return scenarios
        
    def generate_scenario_report(self) -> str:
        """시나리오 리포트 생성"""
        report = []
        report.append("🎯 컵 도달 태스크 - 8가지 시나리오 구성")
        report.append("=" * 60)
        report.append("📋 목표: 작은 박스 장애물을 피해 컵 앞에 도달하기")
        report.append("🔧 변수: 장애물 개수 × 배치 × 경로 = 2 × 2 × 2 = 8가지")
        report.append("")
        
        # 시나리오별 상세 정보
        for i, scenario in enumerate(self.scenarios, 1):
            report.append(f"📌 시나리오 {i}: {scenario.name}")
            report.append(f"   ID: {scenario.id}")
            report.append(f"   설명: {scenario.description}")
            report.append(f"   목표 에피소드: {scenario.target_episodes}개")
            report.append(f"   경로 예시:")
            for line in scenario.example_path.strip().split('\n'):
                if line.strip():
                    report.append(f"   {line}")
            report.append("")
            
        # 전체 통계
        total_episodes = sum(s.target_episodes for s in self.scenarios)
        report.append("📊 전체 목표:")
        report.append(f"   총 에피소드: {total_episodes}개")
        report.append(f"   시나리오당 평균: {total_episodes//len(self.scenarios)}개")
        report.append("")
        
        # 키 매핑 제안
        report.append("🎮 Data Collector 키 매핑 제안:")
        for i, scenario in enumerate(self.scenarios, 1):
            if i <= 8:  # 숫자 키 1-8 사용
                report.append(f"   {i}키: {scenario.name}")
        report.append("")
        
        # 데이터 균형 전략
        report.append("⚖️ 데이터 균형 전략:")
        report.append("   1️⃣ 장애물 개수 균형: 1개(60개) vs 2개(60개)")
        report.append("   2️⃣ 배치 방향 균형: 세로(60개) vs 가로(60개)")  
        report.append("   3️⃣ 경로 선택 균형: 왼쪽(60개) vs 오른쪽(60개)")
        report.append("   4️⃣ 각 시나리오: 15개씩 균등 수집")
        report.append("")
        
        # 예상 학습 효과
        report.append("🤖 예상 VLA 학습 효과:")
        report.append("   ✅ 장애물 개수 인식: 1개 vs 2개 상황 구분")
        report.append("   ✅ 공간 배치 이해: 세로 vs 가로 배치 대응")
        report.append("   ✅ 경로 최적화: 상황별 최적 경로 선택")
        report.append("   ✅ 일반화 능력: 새로운 배치에도 적응")
        
        return "\n".join(report)
        
    def save_scenarios(self, filepath: str):
        """시나리오 정보를 JSON으로 저장"""
        data = {
            "task_name": "cup_reaching_with_obstacles",
            "total_scenarios": len(self.scenarios),
            "total_target_episodes": sum(s.target_episodes for s in self.scenarios),
            "scenarios": [
                {
                    "id": s.id,
                    "name": s.name, 
                    "description": s.description,
                    "obstacle_count": s.obstacle_count,
                    "obstacle_layout": s.obstacle_layout,
                    "preferred_path": s.preferred_path,
                    "target_episodes": s.target_episodes,
                    "example_path": s.example_path
                }
                for s in self.scenarios
            ]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
    def get_wasd_path_examples(self) -> str:
        """WASD 키로 경로 예시 생성"""
        examples = []
        examples.append("🎮 WASD 경로 예시:")
        examples.append("=" * 40)
        
        # 1박스 세로 왼쪽 경로
        examples.append("📌 1박스-세로-왼쪽경로:")
        examples.append("   시작 → W W W → A A → W W → D D → 컵")
        examples.append("   (전진 → 왼쪽우회 → 전진 → 오른쪽복귀)")
        examples.append("")
        
        # 1박스 세로 오른쪽 경로  
        examples.append("📌 1박스-세로-오른쪽경로:")
        examples.append("   시작 → W W → D D → W W W → A A → 컵")
        examples.append("   (전진 → 오른쪽우회 → 전진 → 왼쪽복귀)")
        examples.append("")
        
        # 2박스 가로 왼쪽 경로
        examples.append("📌 2박스-가로-왼쪽경로:")
        examples.append("   시작 → W → A A A → W W → D D D → 컵") 
        examples.append("   (전진 → 왼쪽 크게 우회 → 전진 → 오른쪽 복귀)")
        examples.append("")
        
        # 복잡한 경로 예시
        examples.append("📌 복잡한 회피 패턴:")
        examples.append("   시작 → W W → A → W → D → W → A A → W → D → 컵")
        examples.append("   (지그재그 회피 패턴)")
        
        return "\n".join(examples)

def main():
    """메인 실행 함수"""
    strategy = CupReachingStrategy()
    
    # 리포트 생성 및 출력
    report = strategy.generate_scenario_report()
    print(report)
    
    print("\n" + "=" * 60)
    
    # WASD 경로 예시
    wasd_examples = strategy.get_wasd_path_examples()
    print(wasd_examples)
    
    # JSON 저장
    strategy.save_scenarios("cup_reaching_scenarios.json")
    print(f"\n💾 시나리오 정보가 저장되었습니다: cup_reaching_scenarios.json")

if __name__ == "__main__":
    main()
