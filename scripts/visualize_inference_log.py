#!/usr/bin/env python3
import json
import argparse
import matplotlib.pyplot as plt
import os
import sys

def main():
    parser = argparse.ArgumentParser(description='Visualize Inference Log JSON')
    parser.add_argument('json_path', type=str, help='Path to the JSON log file')
    args = parser.parse_args()
    
    if not os.path.exists(args.json_path):
        print(f"Error: File not found: {args.json_path}")
        sys.exit(1)
        
    print(f"Loading data from {args.json_path}...")
    with open(args.json_path, 'r') as f:
        data = json.load(f)
        
    timestamp = data.get("timestamp", "Unknown Time")
    mem_stats = data.get("memory_stats", {})
    traj_stats = data.get("trajectory_stats", {})
    
    # 데이터 추출
    steps = mem_stats.get('step', [])
    cpu = mem_stats.get('cpu', [])
    gpu = mem_stats.get('gpu', [])
    
    traj_step = traj_stats.get('step', [])
    traj_x = traj_stats.get('x', [])
    traj_y = traj_stats.get('y', [])
    act_x = traj_stats.get('action_x', [])
    act_y = traj_stats.get('action_y', [])
    
    # 그래프 생성
    fig = plt.figure(figsize=(15, 12))
    fig.suptitle(f"Mobile VLA Inference Analysis ({timestamp})", fontsize=16)
    
    # 1. 2D Trajectory (Top-Down View)
    ax1 = fig.add_subplot(2, 2, 1)
    if traj_x and traj_y:
        ax1.plot(traj_x, traj_y, marker='o', label='Path', color='blue', linewidth=2)
        ax1.scatter(traj_x[0], traj_y[0], color='green', s=150, label='Start (0,0)', zorder=5)
        ax1.scatter(traj_x[-1], traj_y[-1], color='red', s=150, label='End', zorder=5)
        
        # 방향 표시 (화살표)
        for i in range(len(traj_x)-1):
            if i % 3 == 0:  # 너무 조밀하지 않게
                ax1.annotate("", xy=(traj_x[i+1], traj_y[i+1]), xytext=(traj_x[i], traj_y[i]),
                            arrowprops=dict(arrowstyle="->", color="gray", alpha=0.5))
        
        # 스텝 번호
        for i, txt in enumerate(traj_step):
            if i % 2 == 0:
                ax1.annotate(str(txt), (traj_x[i+1], traj_y[i+1]), fontsize=9, 
                            xytext=(5, 5), textcoords='offset points')
                
    ax1.set_title("Robot 2D Trajectory (Top-Down)")
    ax1.set_xlabel("Linear X (Forward + / Backward -)")
    ax1.set_ylabel("Linear Y (Left + / Right -)")
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.grid(True)
    ax1.legend()
    ax1.axis('equal')

    # 2. Resource Usage
    ax2 = fig.add_subplot(2, 2, 2)
    if steps:
        ax2.plot(steps, cpu, label='CPU RAM (GB)', marker='o', color='purple')
        ax2.plot(steps, gpu, label='GPU VRAM (GB)', marker='s', color='orange')
    ax2.set_title("System Resource Usage per Frame")
    ax2.set_xlabel("Frame Step")
    ax2.set_ylabel("Memory (GB)")
    ax2.grid(True)
    ax2.legend()

    # 3. Y-Displacement over Time (사용자 요청: Y값이 0부터 시작하여 프레임까지)
    ax3 = fig.add_subplot(2, 2, 3)
    if traj_y:
        # traj_y는 초기값 0.0을 포함하므로 길이가 step+1임
        # X축을 0~18 프레임으로 생성
        frames = list(range(len(traj_y)))
        ax3.plot(frames, traj_y, marker='o', color='crimson', linewidth=2)
        ax3.fill_between(frames, traj_y, 0, alpha=0.1, color='crimson')
        
    ax3.set_title("Lateral Position (Y) over Time")
    ax3.set_xlabel("Frame Sequence (0 = Start)")
    ax3.set_ylabel("Cumulative Y Position (Left+/Right-)")
    ax3.axhline(0, color='gray', linestyle='--')
    ax3.grid(True)
    
    # 4. X-Displacement over Time
    ax4 = fig.add_subplot(2, 2, 4)
    if traj_x:
        frames = list(range(len(traj_x)))
        ax4.plot(frames, traj_x, marker='o', color='teal', linewidth=2)
        ax4.fill_between(frames, traj_x, 0, alpha=0.1, color='teal')
        
    ax4.set_title("Forward Position (X) over Time")
    ax4.set_xlabel("Frame Sequence (0 = Start)")
    ax4.set_ylabel("Cumulative X Position (Forward+/Back-)")
    ax4.axhline(0, color='gray', linestyle='--')
    ax4.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save
    output_path = args.json_path.replace('.json', '_report.png')
    plt.savefig(output_path)
    print(f"✅ Graph saved to: {output_path}")
    
if __name__ == "__main__":
    main()
