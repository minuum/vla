#!/usr/bin/env python3
"""
액션 파싱 시스템 - VLM 출력을 실제 로봇 액션으로 변환
"""

import re
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

class ActionType(Enum):
    """로봇 액션 타입 정의"""
    MOVE_TO = "move_to"
    PICK_UP = "pick_up"
    PLACE_DOWN = "place_down"
    ROTATE = "rotate"
    OPEN = "open"
    CLOSE = "close"
    PUSH = "push"
    PULL = "pull"
    WAIT = "wait"
    STOP = "stop"

@dataclass
class RobotAction:
    """로봇 액션 데이터 구조"""
    action_type: ActionType
    target_object: Optional[str] = None
    target_position: Optional[Tuple[float, float, float]] = None
    target_orientation: Optional[Tuple[float, float, float, float]] = None
    parameters: Optional[Dict[str, Any]] = None
    confidence: float = 1.0
    execution_time: float = 5.0  # 예상 실행 시간 (초)

class ActionParser:
    """VLM 출력을 로봇 액션으로 파싱하는 클래스"""
    
    def __init__(self):
        # 액션 키워드 매핑
        self.action_keywords = {
            ActionType.MOVE_TO: ["move", "go", "navigate", "travel", "approach"],
            ActionType.PICK_UP: ["pick", "grab", "grasp", "take", "lift"],
            ActionType.PLACE_DOWN: ["place", "put", "drop", "set", "release"],
            ActionType.ROTATE: ["rotate", "turn", "spin", "orient"],
            ActionType.OPEN: ["open", "unlock", "unfold"],
            ActionType.CLOSE: ["close", "shut", "lock", "fold"],
            ActionType.PUSH: ["push", "press", "shove"],
            ActionType.PULL: ["pull", "drag", "draw"],
            ActionType.WAIT: ["wait", "pause", "hold", "stay"],
            ActionType.STOP: ["stop", "halt", "cease", "end"]
        }
        
        # 객체 위치 매핑 (이미지 좌표 -> 3D 월드 좌표)
        self.workspace_bounds = {
            'x_min': -0.5, 'x_max': 0.5,
            'y_min': -0.3, 'y_max': 0.3,
            'z_min': 0.0, 'z_max': 0.3
        }
        
    def parse_vlm_output(self, vlm_text: str, detections: List[Dict]) -> List[RobotAction]:
        """
        VLM 텍스트 출력과 객체 탐지 결과를 로봇 액션으로 변환
        
        Args:
            vlm_text: VLM이 생성한 텍스트 설명
            detections: 객체 탐지 결과 (바운딩 박스, 레이블 포함)
            
        Returns:
            List[RobotAction]: 실행 가능한 로봇 액션 리스트
        """
        actions = []
        
        # 1. 텍스트에서 액션 의도 추출
        action_intents = self._extract_action_intents(vlm_text)
        
        # 2. 각 액션 의도를 구체적인 로봇 액션으로 변환
        for intent in action_intents:
            robot_action = self._convert_intent_to_action(intent, detections)
            if robot_action:
                actions.append(robot_action)
                
        return actions
    
    def _extract_action_intents(self, text: str) -> List[Dict]:
        """텍스트에서 액션 의도를 추출"""
        intents = []
        text_lower = text.lower()
        
        # 문장 단위로 분리
        sentences = re.split(r'[.!?;]', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # 각 액션 타입에 대해 키워드 매칭
            for action_type, keywords in self.action_keywords.items():
                for keyword in keywords:
                    if keyword in sentence.lower():
                        # 대상 객체 추출
                        target_object = self._extract_target_object(sentence)
                        
                        intent = {
                            'action_type': action_type,
                            'target_object': target_object,
                            'original_text': sentence,
                            'confidence': self._calculate_confidence(sentence, keyword)
                        }
                        intents.append(intent)
                        break
                        
        return intents
    
    def _extract_target_object(self, sentence: str) -> Optional[str]:
        """문장에서 대상 객체를 추출"""
        # 일반적인 객체 명사들
        common_objects = [
            "cup", "bottle", "book", "phone", "key", "pen", "paper",
            "box", "bag", "chair", "table", "door", "window", "drawer",
            "apple", "banana", "orange", "water", "coffee", "tea"
        ]
        
        sentence_lower = sentence.lower()
        for obj in common_objects:
            if obj in sentence_lower:
                return obj
                
        # 명사 패턴 매칭 (the/a/an 다음의 단어)
        noun_pattern = r'\b(?:the|a|an)\s+(\w+)'
        matches = re.findall(noun_pattern, sentence_lower)
        if matches:
            return matches[0]
            
        return None
    
    def _calculate_confidence(self, sentence: str, keyword: str) -> float:
        """액션 의도의 신뢰도 계산"""
        base_confidence = 0.7
        
        # 키워드가 문장 시작 부분에 있으면 신뢰도 증가
        if sentence.lower().startswith(keyword):
            base_confidence += 0.2
            
        # 명확한 대상 객체가 있으면 신뢰도 증가
        if self._extract_target_object(sentence):
            base_confidence += 0.1
            
        return min(base_confidence, 1.0)
    
    def _convert_intent_to_action(self, intent: Dict, detections: List[Dict]) -> Optional[RobotAction]:
        """액션 의도를 구체적인 로봇 액션으로 변환"""
        action_type = intent['action_type']
        target_object = intent['target_object']
        
        # 탐지된 객체 중에서 대상 객체 찾기
        target_detection = None
        if target_object:
            target_detection = self._find_matching_detection(target_object, detections)
        
        # 액션 타입별 처리
        if action_type == ActionType.PICK_UP:
            return self._create_pickup_action(target_detection, intent)
        elif action_type == ActionType.PLACE_DOWN:
            return self._create_place_action(target_detection, intent)
        elif action_type == ActionType.MOVE_TO:
            return self._create_move_action(target_detection, intent)
        elif action_type == ActionType.ROTATE:
            return self._create_rotate_action(target_detection, intent)
        else:
            # 기본 액션 생성
            return RobotAction(
                action_type=action_type,
                target_object=target_object,
                confidence=intent['confidence']
            )
    
    def _find_matching_detection(self, target_object: str, detections: List[Dict]) -> Optional[Dict]:
        """대상 객체와 매칭되는 탐지 결과 찾기"""
        target_lower = target_object.lower()
        
        for detection in detections:
            label = detection.get('label', '').lower()
            if target_lower in label or label in target_lower:
                return detection
                
        return None
    
    def _create_pickup_action(self, detection: Optional[Dict], intent: Dict) -> RobotAction:
        """픽업 액션 생성"""
        if detection:
            # 바운딩 박스 중심점을 3D 좌표로 변환
            bbox = detection['scaled_box_pixels']
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            # 이미지 좌표를 로봇 좌표계로 변환 (예시)
            world_x = self._pixel_to_world_x(center_x)
            world_y = self._pixel_to_world_y(center_y)
            world_z = 0.05  # 테이블 위 높이
            
            return RobotAction(
                action_type=ActionType.PICK_UP,
                target_object=intent['target_object'],
                target_position=(world_x, world_y, world_z),
                parameters={'approach_height': 0.15, 'grip_force': 0.5},
                confidence=intent['confidence'],
                execution_time=8.0
            )
        else:
            # 객체를 찾지 못한 경우 탐색 액션
            return RobotAction(
                action_type=ActionType.MOVE_TO,
                target_object=intent['target_object'],
                parameters={'search_mode': True},
                confidence=intent['confidence'] * 0.5,
                execution_time=10.0
            )
    
    def _create_place_action(self, detection: Optional[Dict], intent: Dict) -> RobotAction:
        """배치 액션 생성"""
        # 기본 배치 위치 (테이블 중앙)
        place_position = (0.0, 0.0, 0.05)
        
        if detection:
            # 탐지된 객체 근처에 배치
            bbox = detection['scaled_box_pixels']
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            world_x = self._pixel_to_world_x(center_x) + 0.1  # 약간 옆에 배치
            world_y = self._pixel_to_world_y(center_y)
            place_position = (world_x, world_y, 0.05)
        
        return RobotAction(
            action_type=ActionType.PLACE_DOWN,
            target_object=intent['target_object'],
            target_position=place_position,
            parameters={'release_height': 0.02},
            confidence=intent['confidence'],
            execution_time=5.0
        )
    
    def _create_move_action(self, detection: Optional[Dict], intent: Dict) -> RobotAction:
        """이동 액션 생성"""
        if detection:
            bbox = detection['scaled_box_pixels']
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            world_x = self._pixel_to_world_x(center_x)
            world_y = self._pixel_to_world_y(center_y)
            
            return RobotAction(
                action_type=ActionType.MOVE_TO,
                target_object=intent['target_object'],
                target_position=(world_x, world_y, 0.2),  # 안전한 높이
                confidence=intent['confidence'],
                execution_time=6.0
            )
        else:
            # 기본 위치로 이동
            return RobotAction(
                action_type=ActionType.MOVE_TO,
                target_position=(0.0, 0.0, 0.2),
                confidence=intent['confidence'] * 0.7,
                execution_time=4.0
            )
    
    def _create_rotate_action(self, detection: Optional[Dict], intent: Dict) -> RobotAction:
        """회전 액션 생성"""
        # 기본 90도 회전
        rotation_angle = np.pi / 2
        
        return RobotAction(
            action_type=ActionType.ROTATE,
            target_object=intent['target_object'],
            target_orientation=(0, 0, rotation_angle, 1),  # 쿼터니언
            parameters={'rotation_speed': 0.5},
            confidence=intent['confidence'],
            execution_time=3.0
        )
    
    def _pixel_to_world_x(self, pixel_x: float, image_width: int = 640) -> float:
        """픽셀 X 좌표를 월드 X 좌표로 변환"""
        normalized_x = pixel_x / image_width
        return self.workspace_bounds['x_min'] + normalized_x * (
            self.workspace_bounds['x_max'] - self.workspace_bounds['x_min']
        )
    
    def _pixel_to_world_y(self, pixel_y: float, image_height: int = 480) -> float:
        """픽셀 Y 좌표를 월드 Y 좌표로 변환"""
        normalized_y = pixel_y / image_height
        return self.workspace_bounds['y_max'] - normalized_y * (
            self.workspace_bounds['y_max'] - self.workspace_bounds['y_min']
        )
    
    def actions_to_json(self, actions: List[RobotAction]) -> str:
        """액션 리스트를 JSON 문자열로 변환"""
        action_dicts = []
        for action in actions:
            action_dict = {
                'action_type': action.action_type.value,
                'target_object': action.target_object,
                'target_position': action.target_position,
                'target_orientation': action.target_orientation,
                'parameters': action.parameters,
                'confidence': action.confidence,
                'execution_time': action.execution_time
            }
            action_dicts.append(action_dict)
        
        return json.dumps(action_dicts, indent=2)

# 사용 예시
if __name__ == "__main__":
    parser = ActionParser()
    
    # 예시 VLM 출력
    vlm_text = "Pick up the red cup and place it on the table."
    
    # 예시 객체 탐지 결과
    detections = [
        {
            'label': 'cup',
            'scaled_box_pixels': [100, 150, 200, 250],
            'confidence': 0.9
        }
    ]
    
    # 액션 파싱
    actions = parser.parse_vlm_output(vlm_text, detections)
    
    # 결과 출력
    print("파싱된 액션들:")
    for i, action in enumerate(actions):
        print(f"{i+1}. {action.action_type.value}: {action.target_object}")
        print(f"   위치: {action.target_position}")
        print(f"   신뢰도: {action.confidence:.2f}")
        print(f"   예상 실행 시간: {action.execution_time}초")
        print() 