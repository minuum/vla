#!/usr/bin/env python3
"""
데이터셋 통계 분석 및 시각화
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

sns.set_style("whitegrid")

def analyze_dataset():
    """500개 episodes 통계 분석"""
    
    dataset_dir = Path("/home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset")
    h5_files = sorted(dataset_dir.glob("*.h5"))
    
    print(f"📂 Dataset: {len(h5_files)} episodes (Total in directory)")
    
    # 500개 데이터셋 필터링 (202512*)
    h5_files = [f for f in h5_files if "202512" in f.name]
    print(f"Dataset filtered: {len(h5_files)} episodes (202512* only)")
    
    if len(h5_files) != 500:
        print(f"⚠️ 경고: 필터링된 파일 수가 500개가 아닙니다 ({len(h5_files)}개).")
    
    # 통계 수집
    stats = {
        "total_episodes": len(h5_files),
        "left_episodes": 0,
        "right_episodes": 0,
        "episode_lengths": [],
        "actions_linear_x": [],
        "actions_linear_y": [],
        "left_actions": {"linear_x": [], "linear_y": []},
        "right_actions": {"linear_x": [], "linear_y": []}
    }
    
    for h5_file in h5_files:
        # Left/Right 분류
        direction = "left" if "left" in h5_file.name else "right"
        
        if direction == "left":
            stats["left_episodes"] += 1
        else:
            stats["right_episodes"] += 1
        
        try:
            with h5py.File(h5_file, 'r') as f:
                # Episode 길이 ('actions' 키 사용)
                if 'actions' in f:
                    actions = f['actions'][:]
                    episode_length = len(actions)
                    stats["episode_lengths"].append(episode_length)
                    
                    # Actions 수집
                    if len(actions) > 0:
                        linear_x = actions[:, 0]
                        linear_y = actions[:, 1]
                        
                        stats["actions_linear_x"].extend(linear_x)
                        stats["actions_linear_y"].extend(linear_y)
                        
                        if direction == "left":
                            stats["left_actions"]["linear_x"].extend(linear_x)
                            stats["left_actions"]["linear_y"].extend(linear_y)
                        else:
                            stats["right_actions"]["linear_x"].extend(linear_x)
                            stats["right_actions"]["linear_y"].extend(linear_y)
        except Exception as e:
            print(f"⚠️  {h5_file.name}: {e}")
    
    # 통계 계산
    stats["episode_lengths"] = np.array(stats["episode_lengths"])
    stats["actions_linear_x"] = np.array(stats["actions_linear_x"])
    stats["actions_linear_y"] = np.array(stats["actions_linear_y"])
    
    if len(stats["episode_lengths"]) == 0:
        print("\n❌ 에러: episode_lengths가 비어 있습니다. 데이터셋을 확인하세요.")
        return None
    
    summary = {
        "total_episodes": stats["total_episodes"],
        "left_episodes": stats["left_episodes"],
        "right_episodes": stats["right_episodes"],
        "episode_length": {
            "mean": float(np.mean(stats["episode_lengths"])),
            "std": float(np.std(stats["episode_lengths"])),
            "min": int(np.min(stats["episode_lengths"])),
            "max": int(np.max(stats["episode_lengths"]))
        },
        "actions": {
            "linear_x": {
                "mean": float(np.mean(stats["actions_linear_x"])),
                "std": float(np.std(stats["actions_linear_x"])),
                "min": float(np.min(stats["actions_linear_x"])),
                "max": float(np.max(stats["actions_linear_x"]))
            },
            "linear_y": {
                "mean": float(np.mean(stats["actions_linear_y"])),
                "std": float(np.std(stats["actions_linear_y"])),
                "min": float(np.min(stats["actions_linear_y"])),
                "max": float(np.max(stats["actions_linear_y"]))
            }
        }
    }
    
    # 출력
    print("\n📊 Dataset Statistics:")
    print(f"  Total: {summary['total_episodes']} episodes")
    print(f"  Left: {summary['left_episodes']}, Right: {summary['right_episodes']}")
    print(f"\n  Episode Length:")
    print(f"    Mean: {summary['episode_length']['mean']:.1f} ± {summary['episode_length']['std']:.1f}")
    print(f"    Range: [{summary['episode_length']['min']}, {summary['episode_length']['max']}]")
    print(f"\n  Actions (linear_x):")
    print(f"    Mean: {summary['actions']['linear_x']['mean']:.3f} ± {summary['actions']['linear_x']['std']:.3f}")
    print(f"    Range: [{summary['actions']['linear_x']['min']:.3f}, {summary['actions']['linear_x']['max']:.3f}]")
    print(f"\n  Actions (linear_y):")
    print(f"    Mean: {summary['actions']['linear_y']['mean']:.3f} ± {summary['actions']['linear_y']['std']:.3f}")
    print(f"    Range: [{summary['actions']['linear_y']['min']:.3f}, {summary['actions']['linear_y']['max']:.3f}]")
    
    # JSON 저장
    output_file = Path("/home/billy/25-1kp/vla/docs/dataset_statistics.json")
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n✅ 통계 저장: {output_file}")
    
    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Episode Length Distribution
    axes[0, 0].hist(stats["episode_lengths"], bins=30, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel("Episode Length (frames)")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].set_title("Episode Length Distribution")
    axes[0, 0].axvline(summary['episode_length']['mean'], color='red', linestyle='--', label=f"Mean: {summary['episode_length']['mean']:.1f}")
    axes[0, 0].legend()
    
    # 2. Action Distribution (linear_x)
    axes[0, 1].hist(stats["actions_linear_x"], bins=50, edgecolor='black', alpha=0.7, color='green')
    axes[0, 1].set_xlabel("linear_x (m/s)")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_title("linear_x Distribution")
    axes[0, 1].axvline(summary['actions']['linear_x']['mean'], color='red', linestyle='--', label=f"Mean: {summary['actions']['linear_x']['mean']:.3f}")
    axes[0, 1].legend()
    
    # 3. Action Distribution (linear_y)
    axes[1, 0].hist(stats["actions_linear_y"], bins=50, edgecolor='black', alpha=0.7, color='orange')
    axes[1, 0].set_xlabel("linear_y (rad/s)")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].set_title("linear_y Distribution")
    axes[1, 0].axvline(summary['actions']['linear_y']['mean'], color='red', linestyle='--', label=f"Mean: {summary['actions']['linear_y']['mean']:.3f}")
    axes[1, 0].legend()
    
    # 4. Action Scatter (Left vs Right)
    left_x = stats["left_actions"]["linear_x"]
    left_y = stats["left_actions"]["linear_y"]
    right_x = stats["right_actions"]["linear_x"]
    right_y = stats["right_actions"]["linear_y"]
    
    axes[1, 1].scatter(left_x, left_y, alpha=0.3, s=1, label='Left', color='blue')
    axes[1, 1].scatter(right_x, right_y, alpha=0.3, s=1, label='Right', color='red')
    axes[1, 1].set_xlabel("linear_x (m/s)")
    axes[1, 1].set_ylabel("linear_y (rad/s)")
    axes[1, 1].set_title("Action Space (Left vs Right)")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 그래프 저장
    output_fig = Path("/home/billy/25-1kp/vla/docs/figures/dataset_statistics.png")
    output_fig.parent.mkdir(exist_ok=True)
    plt.savefig(output_fig, dpi=300, bbox_inches='tight')
    print(f"✅ 그래프 저장: {output_fig}")
    
    return summary

if __name__ == "__main__":
    print("="*60)
    print("데이터셋 통계 분석")
    print("="*60)
    
    analyze_dataset()
    
    print("\n" + "="*60)
    print("✅ 분석 완료!")
    print("="*60)
