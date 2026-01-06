"""
Instruction Mapping for Mobile VLA

Changed to English instructions (2026-01-07) for Kosmos-2 VLM compatibility.
Matches RoboVLMs_upstream/robovlms/data/mobile_vla_action_dataset.py instruction format.
"""

# English instructions (VLA standard: OpenVLA, RT-2)
SCENARIO_INSTRUCTIONS_EN = {
    "1box_vert_left": "Navigate around the obstacle on the left side and reach the cup",
    "1box_vert_right": "Navigate around the obstacle on the right side and reach the cup",
    "1box_hori_left": "Navigate around the obstacle on the left side and reach the cup",
    "1box_hori_right": "Navigate around the obstacle on the right side and reach the cup",
    "2box_vert_left": "Navigate around the obstacle on the left side and reach the cup",
    "2box_vert_right": "Navigate around the obstacle on the right side and reach the cup",
    "2box_hori_left": "Navigate around the obstacle on the left side and reach the cup",
    "2box_hori_right": "Navigate around the obstacle on the right side and reach the cup",
}

# Robot scenario ID → scenario name mapping
# (Robot node uses ID: '1'=left, '2'=right)
ROBOT_SCENARIO_MAP = {
    '1': "1box_hori_left",   # Left navigation
    '2': "1box_hori_right",  # Right navigation
}

# Default fallback instruction
DEFAULT_INSTRUCTION = "Reach the cup"


def get_instruction_for_scenario(scenario: str) -> str:
    """
    시나리오 문자열로부터 영어 instruction 반환
    
    Args:
        scenario: 'left', 'right', '1box_hori_left' 등
        
    Returns:
        영어 instruction (학습 데이터와 일치)
        
    Examples:
        >>> get_instruction_for_scenario('1box_hori_left')
        'Navigate around the obstacle on the left side and reach the cup'
        >>> get_instruction_for_scenario('left')
        'Navigate around the obstacle on the left side and reach the cup'
    """
    # 직접 매칭 시도
    if scenario in SCENARIO_INSTRUCTIONS_EN:
        return SCENARIO_INSTRUCTIONS_EN[scenario]
    
    # 로봇 ID 변환 시도
    if scenario in ROBOT_SCENARIO_MAP:
        mapped_scenario = ROBOT_SCENARIO_MAP[scenario]
        return SCENARIO_INSTRUCTIONS_EN.get(mapped_scenario, DEFAULT_INSTRUCTION)
    
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
        
    Returns:
        영어 instruction (학습 데이터와 일치)
        
    Examples:
        >>> get_instruction_for_robot_id('1')
        'Navigate around the obstacle on the left side and reach the cup'
    """
    return get_instruction_for_scenario(robot_scenario_id)
