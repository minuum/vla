"""
Instruction Mapping for Mobile VLA

Changed to English instructions (2026-01-07) for Kosmos-2 VLM compatibility.
Matches RoboVLMs_upstream/robovlms/data/mobile_vla_action_dataset.py instruction format.
"""

# English instructions (matching A5000 training dataset format)
# Training data uses simple, concise instructions: "Navigate to the brown pot on the [left/right]"
# Even Cup scenarios were trained with "brown pot" terminology in our V2 dataset.
SCENARIO_INSTRUCTIONS_EN = {
    # Cup Scenarios (1-4) - trained with "brown pot" terminology
    "1box_vert_left": "Navigate to the brown pot on the left",
    "1box_vert_right": "Navigate to the brown pot on the right",
    "1box_hori_left": "Navigate to the brown pot on the left",
    "1box_hori_right": "Navigate to the brown pot on the right",
    "2box_vert_left": "Navigate to the brown pot on the left",
    "2box_vert_right": "Navigate to the brown pot on the right",
    "2box_hori_left": "Navigate to the brown pot on the left",
    "2box_hori_right": "Navigate to the brown pot on the right",
    # Basket Scenarios (5-8) - same format
    "basket_1box_vert_left": "Navigate to the brown pot on the left",
    "basket_1box_vert_right": "Navigate to the brown pot on the right",
    "basket_1box_hori_left": "Navigate to the brown pot on the left",
    "basket_1box_hori_right": "Navigate to the brown pot on the right",
    "basket_2box_vert_left": "Navigate to the brown pot on the left",
    "basket_2box_vert_right": "Navigate to the brown pot on the right",
    "basket_2box_hori_left": "Navigate to the brown pot on the left",
    "basket_2box_hori_right": "Navigate to the brown pot on the right",
}

# Robot scenario ID → scenario name mapping
# (Robot node uses ID: '1'=left, '2'=right)
ROBOT_SCENARIO_MAP = {
    '1': "1box_hori_left",   # Left navigation (Cup)
    '2': "1box_hori_right",  # Right navigation (Cup)
    '3': "1box_vert_left",   # Vert Left (Cup)
    '4': "1box_vert_right",  # Vert Right (Cup)
    '5': "basket_1box_hori_left",   # Left navigation (Basket)
    '6': "basket_1box_hori_right",  # Right navigation (Basket)
    '7': "basket_1box_vert_left",   # Vert Left (Basket)
    '8': "basket_1box_vert_right",  # Vert Right (Basket)
}

# Default fallback instruction
DEFAULT_INSTRUCTION = "Navigate to the target location"


def get_instruction_for_scenario(scenario: str) -> str:
    """
    시나리오 문자열로부터 영어 instruction 반환
    
    Args:
        scenario: 'left', 'right', '1box_hori_left' 등
        
    Returns:
        영어 instruction (학습 데이터와 일치)
    """
    # 직접 매칭 시도
    if scenario in SCENARIO_INSTRUCTIONS_EN:
        return SCENARIO_INSTRUCTIONS_EN[scenario]
    
    # 로봇 ID 변환 시도
    if scenario in ROBOT_SCENARIO_MAP:
        mapped_scenario = ROBOT_SCENARIO_MAP[scenario]
        return SCENARIO_INSTRUCTIONS_EN.get(mapped_scenario, SCENARIO_INSTRUCTIONS_EN.get("1box_hori_left"))
    
    # 'left' / 'right' 간단 문자열 처리
    scenario_lower = scenario.lower()
    if 'left' in scenario_lower:
        return SCENARIO_INSTRUCTIONS_EN['1box_hori_left']
    elif 'right' in scenario_lower:
        return SCENARIO_INSTRUCTIONS_EN['1box_hori_right']
    
    return DEFAULT_INSTRUCTION


def get_instruction_for_robot_id(robot_scenario_id: str) -> str:
    """
    로봇 시나리오 ID로부터 영어 instruction 반환
    
    Args:
        robot_scenario_id: '1' (left) or '2' (right)
    """
    return get_instruction_for_scenario(robot_scenario_id)
