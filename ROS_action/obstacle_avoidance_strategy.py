#!/usr/bin/env python3
"""
VLA 모델 학습을 위한 장애물 회피 데이터 구성 전략 분석 및 제안

현재 상황: 
- 장애물을 왼쪽/오른쪽 두 방향으로 피하는 샘플들이 혼재
- 프레임 18개 데이터 10개 보유 (좋은 품질의 장애물 회피 시나리오로 추정)

질문: 두 선택지 모두 학습하면 모델이 어떻게 행동할까?
"""

import json
import h5py
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

class ObstacleAvoidanceAnalyzer:
    def __init__(self, dataset_path: str = "mobile_vla_dataset"):
        self.dataset_path = Path(dataset_path)
        
    def analyze_action_patterns(self, episode_files: List[str]) -> Dict:
        """에피소드들의 액션 패턴 분석"""
        results = {
            "left_dominant": [],
            "right_dominant": [],
            "mixed_strategy": [],
            "straight_mostly": []
        }
        
        for episode_file in episode_files:
            h5_path = self.dataset_path / f"{episode_file}.h5"
            if not h5_path.exists():
                continue
                
            try:
                with h5py.File(h5_path, 'r') as f:
                    actions = f['actions'][:]
                    # actions shape: [num_frames, 3] -> [linear_x, linear_y, angular_z]
                    
                    # 회전 방향 분석 (angular_z)
                    left_turns = np.sum(actions[:, 2] > 0.1)  # 왼쪽 회전 (양수)
                    right_turns = np.sum(actions[:, 2] < -0.1)  # 오른쪽 회전 (음수)
                    straight = np.sum(np.abs(actions[:, 2]) <= 0.1)  # 직진
                    
                    # 횡이동 분석 (linear_y)
                    left_moves = np.sum(actions[:, 1] > 0.1)  # 왼쪽 이동
                    right_moves = np.sum(actions[:, 1] < -0.1)  # 오른쪽 이동
                    
                    total_frames = len(actions)
                    
                    # 패턴 분류
                    left_ratio = (left_turns + left_moves) / total_frames
                    right_ratio = (right_turns + right_moves) / total_frames
                    
                    episode_info = {
                        "episode": episode_file,
                        "total_frames": total_frames,
                        "left_turns": left_turns,
                        "right_turns": right_turns,
                        "left_moves": left_moves,
                        "right_moves": right_moves,
                        "straight": straight,
                        "left_ratio": left_ratio,
                        "right_ratio": right_ratio
                    }
                    
                    if left_ratio > 0.6:
                        results["left_dominant"].append(episode_info)
                    elif right_ratio > 0.6:
                        results["right_dominant"].append(episode_info)
                    elif left_ratio > 0.2 and right_ratio > 0.2:
                        results["mixed_strategy"].append(episode_info)
                    else:
                        results["straight_mostly"].append(episode_info)
                        
            except Exception as e:
                print(f"⚠️ {episode_file} 분석 실패: {e}")
                
        return results

def main():
    """메인 분석 함수"""
    print("🔍 장애물 회피 전략 분석")
    print("=" * 60)
    
    analyzer = ObstacleAvoidanceAnalyzer()
    
    # 프레임 18개 데이터 분석 (고품질 장애물 회피 시나리오)
    frame_18_episodes = [
        "episode_20250808_074727", "episode_20250808_070428", "episode_20250808_053623",
        "episode_20250808_065843", "episode_20250808_073602", "episode_20250808_063512", 
        "episode_20250808_074409", "episode_20250808_073405", "episode_20250808_074908",
        "episode_20250808_072715"
    ]
    
    print("📊 프레임 18개 데이터 (고품질 샘플) 분석:")
    results = analyzer.analyze_action_patterns(frame_18_episodes)
    
    print(f"\n🔄 분석 결과:")
    print(f"├─ 왼쪽 우세 에피소드: {len(results['left_dominant'])}개")
    print(f"├─ 오른쪽 우세 에피소드: {len(results['right_dominant'])}개") 
    print(f"├─ 혼합 전략 에피소드: {len(results['mixed_strategy'])}개")
    print(f"└─ 직진 위주 에피소드: {len(results['straight_mostly'])}개")
    
    print(f"\n📋 상세 분석:")
    
    for category, episodes in results.items():
        if episodes:
            print(f"\n🏷️ {category.upper()}:")
            for ep in episodes:
                print(f"   • {ep['episode']}: L={ep['left_ratio']:.2f}, R={ep['right_ratio']:.2f} "
                      f"(턴: L{ep['left_turns']}/R{ep['right_turns']}, 횡이동: L{ep['left_moves']}/R{ep['right_moves']})")
    
    print(f"\n" + "=" * 60)
    print("💡 VLA 모델 학습 전략 제안")
    print("=" * 60)
    
    # 전략 제안
    total_left = len(results['left_dominant'])
    total_right = len(results['right_dominant'])
    total_mixed = len(results['mixed_strategy'])
    
    print(f"\n🎯 현재 데이터 분포:")
    print(f"   왼쪽 회피: {total_left}개, 오른쪽 회피: {total_right}개, 혼합: {total_mixed}개")
    
    if total_left > total_right * 2 or total_right > total_left * 2:
        print(f"\n⚠️ 불균형 감지! 한쪽 방향으로 치우쳐져 있습니다.")
        print(f"   → 모델이 특정 방향을 선호할 가능성이 높습니다.")
    
    print(f"\n🚀 추천 전략:")
    
    if total_mixed >= 3:
        print(f"✅ 전략 1: 혼합 데이터 활용")
        print(f"   → 혼합 전략 에피소드가 {total_mixed}개 있어서 다양한 상황 학습 가능")
        print(f"   → 모델이 상황에 따라 최적 경로를 선택하도록 학습")
    
    print(f"\n✅ 전략 2: 컨텍스트 기반 라벨링")
    print(f"   → 각 에피소드에 '목표 방향' 메타데이터 추가")
    print(f"   → 예: 'avoid_left', 'avoid_right', 'shortest_path'")
    
    print(f"\n✅ 전략 3: 균형 맞추기")
    print(f"   → 부족한 방향의 데이터를 추가 수집")
    print(f"   → 목표: 왼쪽 회피 = 오른쪽 회피 = 혼합 전략")
    
    print(f"\n✅ 전략 4: 상황별 에피소드 분리")
    print(f"   → '좁은 통로', '넓은 공간', '다중 장애물' 등으로 시나리오 구분")
    print(f"   → 각 상황별로 최적 회피 전략 학습")
    
    print(f"\n🤖 모델 행동 예측:")
    if total_mixed >= total_left + total_right:
        print(f"   → 다양한 상황을 고려한 의사결정 가능")
        print(f"   → 상황에 맞는 최적 경로 선택")
    elif total_left > total_right * 2:
        print(f"   → 왼쪽 회피를 선호할 가능성 높음")
        print(f"   → 오른쪽 회피 데이터 추가 필요")
    elif total_right > total_left * 2:
        print(f"   → 오른쪽 회피를 선호할 가능성 높음")
        print(f"   → 왼쪽 회피 데이터 추가 필요")
    else:
        print(f"   → 균형잡힌 학습으로 상황별 적응 가능")

if __name__ == "__main__":
    main()
