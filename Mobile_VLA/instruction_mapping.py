"""
Instruction Mapping for Mobile VLA

학습 시 사용된 한국어 instruction과 로봇 시나리오 ID 간 매핑.
RoboVLMs_upstream/robovlms/data/mobile_vla_action_dataset.py의 instruction과 정확히 일치.
"""

# 학습 시 사용된 한국어 instruction (Line 151-160 from dataset loader)
SCENARIO_INSTRUCTIONS_KO = {
    "1box_vert_left": "가장 왼쪽 외곽으로 돌아 컵까지 가세요",
    "1box_vert_right": "가장 오른쪽 외곽으로 돌아 컵까지 가세요",
    "1box_hori_left": "가장 왼쪽 외곽으로 돌아 컵까지 가세요",
    "1box_hori_right": "가장 오른쪽 외곽으로 돌아 컵까지 가세요",
    "2box_vert_left": "가장 왼쪽 외곽으로 돌아 컵까지 가세요",
    "2box_vert_right": "가장 오른쪽 외곽으로 돌아 컵까지 가세요",
    "2box_hori_left": "가장 왼쪽 외곽으로 돌아 컵까지 가세요",
    "2box_hori_right": "가장 오른쪽 외곽으로 돌아 컵까지 가세요",
}

# 로봇 시나리오 ID → 파일명 시나리오 변환
# (로봇 노드에서 사용하는 ID 체계: '1'=left, '2'=right 등)
ROBOT_SCENARIO_MAP = {
    '1': "1box_hori_left",   # Left navigation
    '2': "1box_hori_right",  # Right navigation
}

# Default fallback instruction
DEFAULT_INSTRUCTION = "컵까지 가세요"


def get_instruction_for_scenario(scenario: str) -> str:
    """
    시나리오 문자열로부터 한국어 instruction 반환
    
    Args:
        scenario: 'left', 'right', '1box_hori_left' 등
        
    Returns:
        학습 시 사용된 한국어 instruction
        
    Examples:
        >>> get_instruction_for_scenario('1box_hori_left')
        '가장 왼쪽 외곽으로 돌아 컵까지 가세요'
        >>> get_instruction_for_scenario('left')
        '가장 왼쪽 외곽으로 돌아 컵까지 가세요'
    """
    # 직접 매칭 시도
    if scenario in SCENARIO_INSTRUCTIONS_KO:
        return SCENARIO_INSTRUCTIONS_KO[scenario]
    
    # 로봇 ID 변환 시도
    if scenario in ROBOT_SCENARIO_MAP:
        mapped_scenario = ROBOT_SCENARIO_MAP[scenario]
        return SCENARIO_INSTRUCTIONS_KO.get(mapped_scenario, DEFAULT_INSTRUCTION)
    
    # 'left' / 'right' 간단 문자열 처리
    scenario_lower = scenario.lower()
    if 'left' in scenario_lower:
        return SCENARIO_INSTRUCTIONS_KO['1box_hori_left']
    elif 'right' in scenario_lower:
        return SCENARIO_INSTRUCTIONS_KO['1box_hori_right']
    
    return DEFAULT_INSTRUCTION


def get_instruction_for_robot_id(robot_scenario_id: str) -> str:
    """
    로봇 시나리오 ID로부터 한국어 instruction 반환
    
    Args:
        robot_scenario_id: '1' (left) or '2' (right)
        
    Returns:
        학습 시 사용된 한국어 instruction
        
    Examples:
        >>> get_instruction_for_robot_id('1')
        '가장 왼쪽 외곽으로 돌아 컵까지 가세요'
    """
    return get_instruction_for_scenario(robot_scenario_id)
