#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import sys, tty, termios
import time
import threading
from typing import Dict, Optional

from geometry_msgs.msg import Twist
from std_msgs.msg import String, Float32MultiArray
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

try:
    from pop.driving import Driving
    ROBOT_AVAILABLE = True
except ImportError:
    print("Warning: pop.driving not available. Using simulation mode.")
    ROBOT_AVAILABLE = False

class RobotControlNode(Node):
    def __init__(self):
        super().__init__('robot_control_node')
        
        # 제어 모드
        self.control_mode = "manual"  # "manual", "vla", "hybrid"
        self.manual_priority = True   # 수동 입력 우선
        
        # 액션 매핑
        self.WASD_TO_ACTION = {
            'w': {"linear_x": 1.15, "linear_y": 0.0, "angular_z": 0.0},
            'a': {"linear_x": 0.0, "linear_y": 1.15, "angular_z": 0.0},
            's': {"linear_x": -1.15, "linear_y": 0.0, "angular_z": 0.0},
            'd': {"linear_x": 0.0, "linear_y": -1.15, "angular_z": 0.0},
            'q': {"linear_x": 1.15, "linear_y": 1.15, "angular_z": 0.0},
            'e': {"linear_x": 1.15, "linear_y": -1.15, "angular_z": 0.0},
            'z': {"linear_x": -1.15, "linear_y": 1.15, "angular_z": 0.0},
            'c': {"linear_x": -1.15, "linear_y": -1.15, "angular_z": 0.0},
            'r': {"linear_x": 0.0, "linear_y": 0.0, "angular_z": 1.15},
            't': {"linear_x": 0.0, "linear_y": 0.0, "angular_z": -1.15},
            ' ': {"linear_x": 0.0, "linear_y": 0.0, "angular_z": 0.0}
        }
        
        self.STOP_ACTION = {"linear_x": 0.0, "linear_y": 0.0, "angular_z": 0.0}
        
        # 현재 상태
        self.current_action = self.STOP_ACTION.copy()
        self.last_vla_action = None
        self.last_vla_confidence = 0.0
        self.movement_timer = None
        
        # 로봇 제어
        if ROBOT_AVAILABLE:
            self.driver = Driving()
            self.throttle = 50
        else:
            self.driver = None
            
        # ROS 설정
        self.setup_ros_components()
        
        # 키보드 입력 스레드
        self.keyboard_thread = threading.Thread(target=self.keyboard_loop)
        self.keyboard_thread.daemon = True
        self.keyboard_thread.start()
        
        self.get_logger().info("🤖 로봇 제어 노드 시작!")
        self.get_logger().info("📋 제어 모드:")
        self.get_logger().info("   M: 수동 모드 (WASD)")
        self.get_logger().info("   V: VLA 자동 모드")
        self.get_logger().info("   H: 하이브리드 모드 (수동 우선)")
        self.get_logger().info("   F/G: 속도 조절")
        self.get_logger().info("   Ctrl+C: 종료")
        
    def setup_ros_components(self):
        """ROS 컴포넌트 설정"""
        # QoS 설정
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=1
        )
        
        # 발행자
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # 구독자들
        self.vla_action_sub = self.create_subscription(
            Twist, 'vla_action_command', self.vla_action_callback, 10
        )
        self.vla_confidence_sub = self.create_subscription(
            Float32MultiArray, 'vla_confidence', self.vla_confidence_callback, 10
        )
        
        self.get_logger().info("✅ ROS 컴포넌트 설정 완료")
        
    def vla_action_callback(self, msg: Twist):
        """VLA 액션 명령 수신"""
        self.last_vla_action = {
            "linear_x": msg.linear.x,
            "linear_y": msg.linear.y,
            "angular_z": msg.angular.z
        }
        
        # 하이브리드 모드에서만 자동 적용
        if self.control_mode == "vla" or (self.control_mode == "hybrid" and not self.manual_priority):
            self.apply_action(self.last_vla_action, "VLA")
            
    def vla_confidence_callback(self, msg: Float32MultiArray):
        """VLA 신뢰도 수신"""
        if msg.data:
            self.last_vla_confidence = msg.data[0]
            
    def keyboard_loop(self):
        """키보드 입력 처리 스레드"""
        while rclpy.ok():
            key = self.get_key()
            self.handle_key_input(key)
            time.sleep(0.01)
            
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
        
    def handle_key_input(self, key: str):
        """키보드 입력 처리"""
        if key == '\x03':  # Ctrl+C
            self.stop_robot()
            sys.exit()
            
        elif key == 'm':
            self.control_mode = "manual"
            self.manual_priority = True
            self.get_logger().info("🎮 수동 제어 모드")
            
        elif key == 'v':
            self.control_mode = "vla"
            self.manual_priority = False
            self.get_logger().info("🤖 VLA 자동 제어 모드")
            
        elif key == 'h':
            self.control_mode = "hybrid"
            self.manual_priority = True
            self.get_logger().info("🔄 하이브리드 모드 (수동 우선)")
            
        elif key == 'f':
            if ROBOT_AVAILABLE:
                self.throttle = max(10, self.throttle - 10)
                self.get_logger().info(f'속도: {self.throttle}%')
                
        elif key == 'g':
            if ROBOT_AVAILABLE:
                self.throttle = min(100, self.throttle + 10)
                self.get_logger().info(f'속도: {self.throttle}%')
                
        elif key in self.WASD_TO_ACTION:
            if self.control_mode == "manual" or (self.control_mode == "hybrid" and self.manual_priority):
                action = self.WASD_TO_ACTION[key]
                self.apply_action(action, f"수동({key.upper()})")
                
                # 타이머로 자동 정지
                if self.movement_timer and self.movement_timer.is_alive():
                    self.movement_timer.cancel()
                    
                self.movement_timer = threading.Timer(0.3, self.stop_movement_timed)
                self.movement_timer.start()
                
    def stop_movement_timed(self):
        """타이머 기반 자동 정지"""
        self.get_logger().info("⏰ 자동 정지")
        self.apply_action(self.STOP_ACTION, "자동정지")
        
    def apply_action(self, action: Dict[str, float], source: str):
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
        
    def calculate_movement_angle(self, action: Dict[str, float]) -> float:
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
        if self.movement_timer and self.movement_timer.is_alive():
            self.movement_timer.cancel()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = RobotControlNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
