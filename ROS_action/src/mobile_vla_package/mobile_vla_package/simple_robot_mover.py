#!/usr/bin/env python3
"""
WASD 수동 로봇 컨트롤러
키보드로 직접 로봇을 조작할 수 있습니다.
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import sys
import tty
import termios
import threading
import time
import numpy as np

try:
    from pop.driving import Driving
    ROBOT_AVAILABLE = True
except ImportError:
    print("Warning: pop.driving not available. Using simulation mode.")
    ROBOT_AVAILABLE = False

class MinimalRobotController(Node):
    """최소 로봇 제어 노드"""
    
    def __init__(self):
        super().__init__('minimal_robot_controller')
        
        # ROS 설정
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # 액션 매핑
        self.WASD_TO_ACTION = {
            'w': {"linear_x": 0.5, "linear_y": 0.0, "angular_z": 0.0},
            'a': {"linear_x": 0.0, "linear_y": 0.5, "angular_z": 0.0},
            's': {"linear_x": -0.5, "linear_y": 0.0, "angular_z": 0.0},
            'd': {"linear_x": 0.0, "linear_y": -0.5, "angular_z": 0.0},
            'q': {"linear_x": 0.3, "linear_y": 0.3, "angular_z": 0.0},
            'e': {"linear_x": 0.3, "linear_y": -0.3, "angular_z": 0.0},
            'z': {"linear_x": -0.3, "linear_y": 0.3, "angular_z": 0.0},
            'c': {"linear_x": -0.3, "linear_y": -0.3, "angular_z": 0.0},
            'r': {"linear_x": 0.0, "linear_y": 0.0, "angular_z": 0.5},
            't': {"linear_x": 0.0, "linear_y": 0.0, "angular_z": -0.5},
            ' ': {"linear_x": 0.0, "linear_y": 0.0, "angular_z": 0.0}
        }
        
        self.STOP_ACTION = {"linear_x": 0.0, "linear_y": 0.0, "angular_z": 0.0}
        self.current_action = self.STOP_ACTION.copy()
        
        # 로봇 제어
        if ROBOT_AVAILABLE:
            self.driver = Driving()
            self.throttle = 50
        else:
            self.driver = None
        
        # 키보드 입력 스레드 시작
        self.keyboard_thread = threading.Thread(target=self.keyboard_loop)
        self.keyboard_thread.daemon = True
        self.keyboard_thread.start()
        
        self.get_logger().info("🤖 최소 로봇 제어 노드 시작")
        self.get_logger().info("📋 조작법: WASD (이동), QEZC (대각선), RT (회전), 스페이스바 (정지)")
    
    def get_key(self) -> str:
        """키보드 입력 읽기"""
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        return ch.lower()
    
    def keyboard_loop(self):
        """키보드 입력 처리 스레드"""
        while rclpy.ok():
            try:
                key = self.get_key()
                self.handle_key_input(key)
                time.sleep(0.01)
            except Exception as e:
                self.get_logger().error(f"❌ 키보드 입력 오류: {e}")
                time.sleep(0.1)
    
    def handle_key_input(self, key: str):
        """키보드 입력 처리"""
        if key == '\x03':  # Ctrl+C
            self.stop_robot()
            sys.exit()
        elif key in self.WASD_TO_ACTION:
            action = self.WASD_TO_ACTION[key]
            self.apply_action(action, f"키보드({key.upper()})")
        else:
            # 알 수 없는 키는 정지
            self.apply_action(self.STOP_ACTION, "정지")
    
    def apply_action(self, action: dict, source: str):
        """액션 적용"""
        self.current_action = action.copy()
        
        # ROS 메시지 발행
        twist = Twist()
        twist.linear.x = float(action["linear_x"])
        twist.linear.y = float(action["linear_y"])
        twist.angular.z = float(action["angular_z"])
        self.cmd_vel_pub.publish(twist)
        
        # 실제 로봇 제어
        if ROBOT_AVAILABLE and self.driver:
            if any(abs(v) > 0.1 for v in action.values()):
                if abs(action["angular_z"]) > 0.1:
                    spin_speed = int(action["angular_z"] * self.throttle)
                    self.driver.spin(spin_speed)
                elif abs(action["linear_x"]) > 0.1 or abs(action["linear_y"]) > 0.1:
                    angle = self.calculate_movement_angle(action)
                    self.driver.move(int(angle), self.throttle)
            else:
                self.driver.stop()
        
        # 로깅
        action_desc = []
        if abs(action['linear_x']) > 0.1:
            action_desc.append(f"전진{action['linear_x']:+.1f}")
        if abs(action['linear_y']) > 0.1:
            action_desc.append(f"횡이동{action['linear_y']:+.1f}")
        if abs(action['angular_z']) > 0.1:
            action_desc.append(f"회전{action['angular_z']:+.1f}")
        if not action_desc:
            action_desc.append("정지")
        
        self.get_logger().info(f"🎯 [{source}] 액션: {', '.join(action_desc)}")
    
    def calculate_movement_angle(self, action: dict) -> float:
        """액션에서 이동 각도 계산"""
        import math
        angle = math.degrees(math.atan2(action["linear_y"], action["linear_x"]))
        if angle < 0:
            angle += 360
        return angle
    
    def stop_robot(self):
        """로봇 정지"""
        self.apply_action(self.STOP_ACTION, "정지")
    
    def destroy_node(self):
        """노드 정리"""
        self.stop_robot()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = MinimalRobotController()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()