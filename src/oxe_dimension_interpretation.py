import numpy as np
import pandas as pd
import json

def interpret_oxe_dimensions():
    """OXE Dataset 차원 해석 및 문서화"""
    
    # 데이터 로드
    data = np.load('cup.npy', allow_pickle=True)
    sample = data[0]
    
    print("=== OXE Dataset 차원 해석 ===")
    print(f"태스크: {sample['task'][0][0]}")
    
    # 액션 분석
    actions = sample['action']
    robot_states = sample['robot_state']
    
    # 액션 차원 해석
    action_interpretation = {
        "action_0": {
            "name": "End-effector X translation",
            "description": "X축 방향 엔드 이펙터 이동 (delta control)",
            "range": f"[{actions[:, 0].min():.4f}, {actions[:, 0].max():.4f}]",
            "unit": "meters/step",
            "evidence": "작은 값(-0.02~0.02), 연속적 변화"
        },
        "action_1": {
            "name": "End-effector Y translation", 
            "description": "Y축 방향 엔드 이펙터 이동 (delta control)",
            "range": f"[{actions[:, 1].min():.4f}, {actions[:, 1].max():.4f}]",
            "unit": "meters/step",
            "evidence": "작은 값(-0.01~0.02), 연속적 변화"
        },
        "action_2": {
            "name": "End-effector Z translation",
            "description": "Z축 방향 엔드 이펙터 이동 (delta control)", 
            "range": f"[{actions[:, 2].min():.4f}, {actions[:, 2].max():.4f}]",
            "unit": "meters/step",
            "evidence": "작은 값(-0.02~0.02), 연속적 변화"
        },
        "action_3": {
            "name": "End-effector Roll rotation",
            "description": "엔드 이펙터 Roll 회전 (delta control)",
            "range": f"[{actions[:, 3].min():.4f}, {actions[:, 3].max():.4f}]",
            "unit": "radians/step", 
            "evidence": "작은 값(-0.02~0.02), angular motion"
        },
        "action_4": {
            "name": "End-effector Pitch rotation",
            "description": "엔드 이펙터 Pitch 회전 (delta control)",
            "range": f"[{actions[:, 4].min():.4f}, {actions[:, 4].max():.4f}]",
            "unit": "radians/step",
            "evidence": "많은 0값 (63.89%), 제한적 움직임"
        },
        "action_5": {
            "name": "End-effector Yaw rotation", 
            "description": "엔드 이펙터 Yaw 회전 (delta control)",
            "range": f"[{actions[:, 5].min():.4f}, {actions[:, 5].max():.4f}]",
            "unit": "radians/step",
            "evidence": "많은 0값 (60.19%), 제한적 움직임"
        },
        "action_6": {
            "name": "Gripper close/open command",
            "description": "그리퍼 닫힘/열림 명령 (discrete)",
            "range": f"[{actions[:, 6].min():.1f}, {actions[:, 6].max():.1f}]",
            "unit": "discrete (-1: 열림, 0: 유지, 1: 닫힘)",
            "evidence": "98.15%가 0값, -1/0/1 discrete values"
        },
        "action_7": {
            "name": "Gripper activation",
            "description": "그리퍼 활성화 신호",
            "range": f"[{actions[:, 7].min():.1f}, {actions[:, 7].max():.1f}]", 
            "unit": "binary (0: 비활성, 1: 활성)",
            "evidence": "99.07%가 0값, binary control signal"
        }
    }
    
    # 로봇 상태 차원 해석
    state_interpretation = {
        "state_0": {
            "name": "Joint 1 angle",
            "description": "첫 번째 관절 각도 (base rotation)",
            "range": f"[{robot_states[:, 0].min():.4f}, {robot_states[:, 0].max():.4f}]",
            "unit": "radians",
            "evidence": "큰 음수값, 관절 각도 범위"
        },
        "state_1": {
            "name": "Joint 2 angle",
            "description": "두 번째 관절 각도 (shoulder)",
            "range": f"[{robot_states[:, 1].min():.4f}, {robot_states[:, 1].max():.4f}]",
            "unit": "radians", 
            "evidence": "음수값, 관절 각도 범위"
        },
        "state_2": {
            "name": "Joint 3 angle", 
            "description": "세 번째 관절 각도 (elbow)",
            "range": f"[{robot_states[:, 2].min():.4f}, {robot_states[:, 2].max():.4f}]",
            "unit": "radians",
            "evidence": "양수값, 관절 각도 범위"
        },
        "state_3": {
            "name": "Joint 4 angle",
            "description": "네 번째 관절 각도 (wrist 1)",
            "range": f"[{robot_states[:, 3].min():.4f}, {robot_states[:, 3].max():.4f}]",
            "unit": "radians",
            "evidence": "음수값, 관절 각도 범위"
        },
        "state_4": {
            "name": "Joint 5 angle",
            "description": "다섯 번째 관절 각도 (wrist 2)", 
            "range": f"[{robot_states[:, 4].min():.4f}, {robot_states[:, 4].max():.4f}]",
            "unit": "radians",
            "evidence": "음수값, 작은 변화량"
        },
        "state_5": {
            "name": "Joint 6 angle",
            "description": "여섯 번째 관절 각도 (wrist 3)",
            "range": f"[{robot_states[:, 5].min():.4f}, {robot_states[:, 5].max():.4f}]",
            "unit": "radians",
            "evidence": "양수값, 큰 변화량"
        },
        "state_6": {
            "name": "Joint 7 angle",
            "description": "일곱 번째 관절 각도 (7DOF arm의 경우)",
            "range": f"[{robot_states[:, 6].min():.4f}, {robot_states[:, 6].max():.4f}]",
            "unit": "radians",
            "evidence": "양수값, 중간 범위"
        },
        "state_7": {
            "name": "Gripper position",
            "description": "그리퍼 위치/개방도",
            "range": f"[{robot_states[:, 7].min():.4f}, {robot_states[:, 7].max():.4f}]",
            "unit": "meters or normalized",
            "evidence": "음수~양수, 그리퍼 동작"
        },
        "state_8": {
            "name": "Gripper velocity", 
            "description": "그리퍼 속도",
            "range": f"[{robot_states[:, 8].min():.4f}, {robot_states[:, 8].max():.4f}]",
            "unit": "meters/sec or normalized",
            "evidence": "음수~양수, 속도 특성"
        },
        "state_9": {
            "name": "End-effector X position",
            "description": "엔드 이펙터 X 좌표",
            "range": f"[{robot_states[:, 9].min():.4f}, {robot_states[:, 9].max():.4f}]",
            "unit": "meters",
            "evidence": "0.7 근처 값, 위치 좌표"
        },
        "state_10": {
            "name": "End-effector Y position",
            "description": "엔드 이펙터 Y 좌표", 
            "range": f"[{robot_states[:, 10].min():.4f}, {robot_states[:, 10].max():.4f}]",
            "unit": "meters",
            "evidence": "0.7 근처 값, 위치 좌표"
        },
        "state_11": {
            "name": "End-effector Z position",
            "description": "엔드 이펙터 Z 좌표",
            "range": f"[{robot_states[:, 11].min():.4f}, {robot_states[:, 11].max():.4f}]",
            "unit": "meters", 
            "evidence": "0 근처 값, 높이 좌표"
        },
        "state_12": {
            "name": "End-effector orientation",
            "description": "엔드 이펙터 방향 (roll/pitch/yaw 중 하나)",
            "range": f"[{robot_states[:, 12].min():.4f}, {robot_states[:, 12].max():.4f}]",
            "unit": "radians",
            "evidence": "작은 양수값, 방향각"
        },
        "state_13": {
            "name": "Gripper state (binary)",
            "description": "그리퍼 상태 (0: 열림, 1: 닫힘)",
            "range": f"[{robot_states[:, 13].min():.1f}, {robot_states[:, 13].max():.1f}]",
            "unit": "binary",
            "evidence": "53.70%가 0값, binary state"
        },
        "state_14": {
            "name": "Gripper lock/unlock",
            "description": "그리퍼 잠금 상태",
            "range": f"[{robot_states[:, 14].min():.1f}, {robot_states[:, 14].max():.1f}]",
            "unit": "binary",
            "evidence": "73.15%가 0값, binary control"
        }
    }
    
    # 종합 해석
    summary = {
        "dataset_info": {
            "type": "OXE (Open X-Embodiment) Robot Demonstration",
            "task": sample['task'][0][0],
            "timesteps": len(actions),
            "robot_type": "7-DOF manipulator with gripper"
        },
        "action_space": {
            "dimension": 8,
            "type": "Continuous delta control + discrete gripper",
            "description": "6D end-effector control (3 translation + 3 rotation) + 2D gripper control",
            "control_frequency": "Estimated ~10-20 Hz based on smooth trajectories"
        },
        "state_space": {
            "dimension": 15, 
            "type": "Mixed continuous joint states + end-effector pose + gripper state",
            "description": "7 joint angles + 2 gripper states + 4 end-effector pose + 2 gripper binary states"
        },
        "interpretations": {
            "actions": action_interpretation,
            "states": state_interpretation
        }
    }
    
    # JSON으로 저장
    with open('oxe_dimension_interpretation.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print("차원 해석 완료!")
    print(f"상세 해석 저장: oxe_dimension_interpretation.json")
    
    # 요약 출력
    print("\n=== 요약 ===")
    print("🤖 액션 8차원:")
    print("  • action_0~2: 엔드 이펙터 XYZ 이동 (delta control)")
    print("  • action_3~5: 엔드 이펙터 Roll/Pitch/Yaw 회전 (delta control)")
    print("  • action_6: 그리퍼 열림/닫힘 명령 (-1/0/1)")
    print("  • action_7: 그리퍼 활성화 신호 (0/1)")
    
    print("\n🔧 로봇 상태 15차원:")
    print("  • state_0~6: 7개 관절 각도 (radians)")
    print("  • state_7~8: 그리퍼 위치/속도")
    print("  • state_9~11: 엔드 이펙터 XYZ 위치 (meters)")
    print("  • state_12: 엔드 이펙터 방향각")
    print("  • state_13~14: 그리퍼 이진 상태 (열림/닫힘, 잠금)")
    
    return summary

if __name__ == "__main__":
    summary = interpret_oxe_dimensions()