#!/usr/bin/env python3
"""
액션 실행기 - 파싱된 액션을 실제 로봇 제어 명령으로 변환
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Twist, PoseStamped, Point
from sensor_msgs.msg import JointState
import json
import time
import threading
from typing import List, Dict, Any
from .action_parser import ActionParser, RobotAction, ActionType

class ActionExecutor(Node):
    """액션 실행기 ROS2 노드"""
    
    def __init__(self):
        super().__init__('action_executor')
        
        # 액션 파서 초기화
        self.action_parser = ActionParser()
        
        # 현재 실행 중인 액션 상태
        self.current_action = None
        self.action_queue = []
        self.is_executing = False
        
        # ROS2 구독자들
        self.vlm_result_sub = self.create_subscription(
            String,
            '/vlm/result',
            self.vlm_result_callback,
            10
        )
        
        self.detection_sub = self.create_subscription(
            String,
            '/object_detection/result',
            self.detection_callback,
            10
        )
        
        # ROS2 발행자들
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.arm_goal_pub = self.create_publisher(PoseStamped, '/arm/goal_pose', 10)
        self.gripper_pub = self.create_publisher(Bool, '/gripper/command', 10)
        self.action_status_pub = self.create_publisher(String, '/action_executor/status', 10)
        
        # 로봇 상태 변수들
        self.robot_pose = {'x': 0.0, 'y': 0.0, 'z': 0.2}
        self.gripper_open = True
        self.arm_busy = False
        
        # 최근 탐지 결과 저장
        self.latest_detections = []
        
        # 액션 실행 스레드
        self.execution_thread = None
        
        self.get_logger().info("액션 실행기가 초기화되었습니다.")
    
    def vlm_result_callback(self, msg):
        """VLM 결과 콜백 - 텍스트 명령을 액션으로 파싱"""
        try:
            vlm_text = msg.data
            self.get_logger().info(f"VLM 결과 수신: {vlm_text}")
            
            # VLM 출력을 액션으로 파싱
            actions = self.action_parser.parse_vlm_output(vlm_text, self.latest_detections)
            
            if actions:
                self.get_logger().info(f"{len(actions)}개의 액션이 파싱되었습니다.")
                
                # 액션 큐에 추가
                for action in actions:
                    self.action_queue.append(action)
                    self.get_logger().info(
                        f"액션 추가: {action.action_type.value} - {action.target_object}"
                    )
                
                # 실행 시작
                if not self.is_executing:
                    self.start_action_execution()
            else:
                self.get_logger().warn("파싱된 액션이 없습니다.")
                
        except Exception as e:
            self.get_logger().error(f"VLM 결과 처리 중 오류: {e}")
    
    def detection_callback(self, msg):
        """객체 탐지 결과 콜백"""
        try:
            detection_data = json.loads(msg.data)
            self.latest_detections = detection_data.get('detections', [])
            self.get_logger().info(f"{len(self.latest_detections)}개의 객체가 탐지되었습니다.")
            
        except Exception as e:
            self.get_logger().error(f"탐지 결과 처리 중 오류: {e}")
    
    def start_action_execution(self):
        """액션 실행 시작"""
        if self.execution_thread and self.execution_thread.is_alive():
            return
            
        self.is_executing = True
        self.execution_thread = threading.Thread(target=self.execute_actions)
        self.execution_thread.start()
    
    def execute_actions(self):
        """액션 큐의 액션들을 순차적으로 실행"""
        while self.action_queue and self.is_executing:
            action = self.action_queue.pop(0)
            self.current_action = action
            
            self.get_logger().info(f"액션 실행 시작: {action.action_type.value}")
            self.publish_status(f"실행 중: {action.action_type.value}")
            
            try:
                success = self.execute_single_action(action)
                
                if success:
                    self.get_logger().info(f"액션 완료: {action.action_type.value}")
                    self.publish_status(f"완료: {action.action_type.value}")
                else:
                    self.get_logger().error(f"액션 실패: {action.action_type.value}")
                    self.publish_status(f"실패: {action.action_type.value}")
                    
            except Exception as e:
                self.get_logger().error(f"액션 실행 중 오류: {e}")
                self.publish_status(f"오류: {action.action_type.value}")
            
            # 액션 간 대기 시간
            time.sleep(1.0)
        
        self.is_executing = False
        self.current_action = None
        self.publish_status("대기 중")
    
    def execute_single_action(self, action: RobotAction) -> bool:
        """단일 액션 실행"""
        action_type = action.action_type
        
        if action_type == ActionType.MOVE_TO:
            return self.execute_move_action(action)
        elif action_type == ActionType.PICK_UP:
            return self.execute_pickup_action(action)
        elif action_type == ActionType.PLACE_DOWN:
            return self.execute_place_action(action)
        elif action_type == ActionType.ROTATE:
            return self.execute_rotate_action(action)
        elif action_type == ActionType.OPEN:
            return self.execute_gripper_action(True)
        elif action_type == ActionType.CLOSE:
            return self.execute_gripper_action(False)
        elif action_type == ActionType.WAIT:
            return self.execute_wait_action(action)
        elif action_type == ActionType.STOP:
            return self.execute_stop_action()
        else:
            self.get_logger().warn(f"지원되지 않는 액션 타입: {action_type}")
            return False
    
    def execute_move_action(self, action: RobotAction) -> bool:
        """이동 액션 실행"""
        if not action.target_position:
            self.get_logger().error("이동 목표 위치가 지정되지 않았습니다.")
            return False
        
        target_x, target_y, target_z = action.target_position
        
        # 팔 위치 제어
        pose_msg = PoseStamped()
        pose_msg.header.frame_id = "base_link"
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.pose.position.x = target_x
        pose_msg.pose.position.y = target_y
        pose_msg.pose.position.z = target_z
        pose_msg.pose.orientation.w = 1.0
        
        self.arm_goal_pub.publish(pose_msg)
        self.get_logger().info(f"팔 이동 명령 전송: ({target_x:.2f}, {target_y:.2f}, {target_z:.2f})")
        
        # 이동 완료 대기 (시뮬레이션)
        time.sleep(action.execution_time)
        
        # 로봇 위치 업데이트
        self.robot_pose['x'] = target_x
        self.robot_pose['y'] = target_y
        self.robot_pose['z'] = target_z
        
        return True
    
    def execute_pickup_action(self, action: RobotAction) -> bool:
        """픽업 액션 실행"""
        if not action.target_position:
            self.get_logger().error("픽업 목표 위치가 지정되지 않았습니다.")
            return False
        
        target_x, target_y, target_z = action.target_position
        approach_height = action.parameters.get('approach_height', 0.15) if action.parameters else 0.15
        
        # 1. 접근 위치로 이동
        approach_action = RobotAction(
            action_type=ActionType.MOVE_TO,
            target_position=(target_x, target_y, target_z + approach_height),
            execution_time=3.0
        )
        if not self.execute_move_action(approach_action):
            return False
        
        # 2. 그리퍼 열기
        if not self.execute_gripper_action(True):
            return False
        
        # 3. 목표 위치로 하강
        pickup_action = RobotAction(
            action_type=ActionType.MOVE_TO,
            target_position=(target_x, target_y, target_z),
            execution_time=2.0
        )
        if not self.execute_move_action(pickup_action):
            return False
        
        # 4. 그리퍼 닫기
        if not self.execute_gripper_action(False):
            return False
        
        # 5. 들어올리기
        lift_action = RobotAction(
            action_type=ActionType.MOVE_TO,
            target_position=(target_x, target_y, target_z + approach_height),
            execution_time=2.0
        )
        if not self.execute_move_action(lift_action):
            return False
        
        self.get_logger().info(f"객체 픽업 완료: {action.target_object}")
        return True
    
    def execute_place_action(self, action: RobotAction) -> bool:
        """배치 액션 실행"""
        if not action.target_position:
            self.get_logger().error("배치 목표 위치가 지정되지 않았습니다.")
            return False
        
        target_x, target_y, target_z = action.target_position
        release_height = action.parameters.get('release_height', 0.02) if action.parameters else 0.02
        
        # 1. 배치 위치 위로 이동
        approach_action = RobotAction(
            action_type=ActionType.MOVE_TO,
            target_position=(target_x, target_y, target_z + 0.1),
            execution_time=3.0
        )
        if not self.execute_move_action(approach_action):
            return False
        
        # 2. 배치 위치로 하강
        place_action = RobotAction(
            action_type=ActionType.MOVE_TO,
            target_position=(target_x, target_y, target_z + release_height),
            execution_time=2.0
        )
        if not self.execute_move_action(place_action):
            return False
        
        # 3. 그리퍼 열기
        if not self.execute_gripper_action(True):
            return False
        
        # 4. 위로 이동
        retreat_action = RobotAction(
            action_type=ActionType.MOVE_TO,
            target_position=(target_x, target_y, target_z + 0.1),
            execution_time=2.0
        )
        if not self.execute_move_action(retreat_action):
            return False
        
        self.get_logger().info(f"객체 배치 완료: {action.target_object}")
        return True
    
    def execute_rotate_action(self, action: RobotAction) -> bool:
        """회전 액션 실행"""
        if not action.target_orientation:
            self.get_logger().error("회전 목표 방향이 지정되지 않았습니다.")
            return False
        
        # 현재 위치에서 방향만 변경
        pose_msg = PoseStamped()
        pose_msg.header.frame_id = "base_link"
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.pose.position.x = self.robot_pose['x']
        pose_msg.pose.position.y = self.robot_pose['y']
        pose_msg.pose.position.z = self.robot_pose['z']
        
        qx, qy, qz, qw = action.target_orientation
        pose_msg.pose.orientation.x = qx
        pose_msg.pose.orientation.y = qy
        pose_msg.pose.orientation.z = qz
        pose_msg.pose.orientation.w = qw
        
        self.arm_goal_pub.publish(pose_msg)
        self.get_logger().info("회전 명령 전송")
        
        time.sleep(action.execution_time)
        return True
    
    def execute_gripper_action(self, open_gripper: bool) -> bool:
        """그리퍼 제어"""
        gripper_msg = Bool()
        gripper_msg.data = open_gripper
        
        self.gripper_pub.publish(gripper_msg)
        
        action_name = "열기" if open_gripper else "닫기"
        self.get_logger().info(f"그리퍼 {action_name} 명령 전송")
        
        # 그리퍼 동작 대기
        time.sleep(1.0)
        
        self.gripper_open = open_gripper
        return True
    
    def execute_wait_action(self, action: RobotAction) -> bool:
        """대기 액션 실행"""
        wait_time = action.execution_time
        self.get_logger().info(f"{wait_time}초 대기")
        time.sleep(wait_time)
        return True
    
    def execute_stop_action(self) -> bool:
        """정지 액션 실행"""
        # 모든 움직임 정지
        twist_msg = Twist()
        self.cmd_vel_pub.publish(twist_msg)
        
        # 액션 큐 클리어
        self.action_queue.clear()
        self.is_executing = False
        
        self.get_logger().info("모든 액션 정지")
        return True
    
    def publish_status(self, status: str):
        """액션 실행 상태 발행"""
        status_msg = String()
        status_msg.data = status
        self.action_status_pub.publish(status_msg)
    
    def emergency_stop(self):
        """비상 정지"""
        self.get_logger().warn("비상 정지 실행")
        self.execute_stop_action()
        self.publish_status("비상 정지")

def main(args=None):
    rclpy.init(args=args)
    
    action_executor = ActionExecutor()
    
    try:
        rclpy.spin(action_executor)
    except KeyboardInterrupt:
        action_executor.get_logger().info("키보드 인터럽트로 종료")
    finally:
        action_executor.emergency_stop()
        action_executor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 