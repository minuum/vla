#!/usr/bin/env python3
"""
로봇 제어 노드
- VLA 추론 결과와 수동 입력 통합
- 안전한 로봇 제어 및 모드 전환
- 실시간 상태 모니터링
"""

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
            depth=10
        )
        
        # 구독자들
        self.vla_action_sub = self.create_subscription(
            Twist, 'vla_action_command', self.vla_action_callback, qos_profile
        )
        self.vla_confidence_sub = self.create_subscription(
            Float32MultiArray, 'vla_confidence', self.vla_confidence_callback, qos_profile
        )
        
        # 발행자
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.control_status_pub = self.create_publisher(String, '/control_status', 10)
        
        # 타이머 (로봇 제어)
        self.control_timer = self.create_timer(0.1, self.control_callback)  # 10Hz
        
        self.get_logger().info("✅ ROS 컴포넌트 설정 완료")
        
    def vla_action_callback(self, msg: Twist):
        """VLA 액션 명령 수신"""
        self.last_vla_action = {
            "linear_x": msg.linear.x,
            "linear_y": msg.linear.y,
            "angular_z": msg.angular.z
        }
        
        if self.control_mode in ["vla", "hybrid"]:
            self.get_logger().info(f"🎯 VLA 액션 수신: {self.last_vla_action}")
            
    def vla_confidence_callback(self, msg: Float32MultiArray):
        """VLA 신뢰도 수신"""
        if msg.data:
            self.last_vla_confidence = msg.data[0]
            
    def get_key(self) -> str:
        """키보드 입력 받기"""
        try:
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(sys.stdin.fileno())
                ch = sys.stdin.read(1)
                return ch.lower()
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        except:
            return ''
            
    def keyboard_loop(self):
        """키보드 입력 처리 루프"""
        while rclpy.ok():
            key = self.get_key()
            
            if key == 'q' and self.control_mode == "manual":
                # 종료
                break
            elif key == 'm':
                # 수동 모드
                self.control_mode = "manual"
                self.get_logger().info("🎮 수동 모드로 전환")
            elif key == 'v':
                # VLA 모드
                self.control_mode = "vla"
                self.get_logger().info("🤖 VLA 자동 모드로 전환")
            elif key == 'h':
                # 하이브리드 모드
                self.control_mode = "hybrid"
                self.get_logger().info("🔄 하이브리드 모드로 전환")
            elif key == 'f':
                # 속도 증가
                self.throttle = min(100, self.throttle + 10)
                self.get_logger().info(f"⚡ 속도 증가: {self.throttle}%")
            elif key == 'g':
                # 속도 감소
                self.throttle = max(10, self.throttle - 10)
                self.get_logger().info(f"🐌 속도 감소: {self.throttle}%")
            elif key in self.WASD_TO_ACTION:
                # WASD 액션
                if self.control_mode in ["manual", "hybrid"]:
                    self.current_action = self.WASD_TO_ACTION[key].copy()
                    self.get_logger().info(f"🎮 수동 액션: {key} → {self.current_action}")
                    
                    # 하이브리드 모드에서 수동 입력 시 VLA 액션 무시
                    if self.control_mode == "hybrid":
                        self.manual_priority = True
                        
            time.sleep(0.01)  # CPU 사용량 감소
            
    def control_callback(self):
        """로봇 제어 콜백"""
        # 현재 모드에 따른 액션 결정
        if self.control_mode == "manual":
            action = self.current_action
        elif self.control_mode == "vla":
            action = self.last_vla_action if self.last_vla_action else self.STOP_ACTION
        elif self.control_mode == "hybrid":
            if self.manual_priority and self.current_action != self.STOP_ACTION:
                action = self.current_action
                # 수동 입력이 없으면 VLA 모드로 전환
                if time.time() - getattr(self, 'last_manual_time', 0) > 2.0:
                    self.manual_priority = False
            else:
                action = self.last_vla_action if self.last_vla_action else self.STOP_ACTION
        else:
            action = self.STOP_ACTION
            
        # 액션 적용
        self.apply_action(action)
        
        # 상태 발행
        self.publish_status()
        
    def apply_action(self, action: Dict[str, float]):
        """액션을 로봇에 적용"""
        if not action:
            return
            
        # 속도 조절 적용
        scaled_action = {
            "linear_x": action["linear_x"] * (self.throttle / 100.0),
            "linear_y": action["linear_y"] * (self.throttle / 100.0),
            "angular_z": action["angular_z"] * (self.throttle / 100.0)
        }
        
        # ROS 메시지 발행
        twist_msg = Twist()
        twist_msg.linear.x = float(scaled_action["linear_x"])
        twist_msg.linear.y = float(scaled_action["linear_y"])
        twist_msg.angular.z = float(scaled_action["angular_z"])
        self.cmd_vel_pub.publish(twist_msg)
        
        # 실제 로봇 제어 (가능한 경우)
        if ROBOT_AVAILABLE and self.driver:
            try:
                # pop.driving API 사용
                self.driver.move(
                    x=scaled_action["linear_x"],
                    y=scaled_action["linear_y"],
                    z=scaled_action["angular_z"]
                )
            except Exception as e:
                self.get_logger().error(f"❌ 로봇 제어 실패: {e}")
                
    def publish_status(self):
        """상태 정보 발행"""
        status_msg = String()
        status_msg.data = f"Mode:{self.control_mode},Throttle:{self.throttle}%"
        self.control_status_pub.publish(status_msg)
        
    def stop_robot(self):
        """로봇 정지"""
        self.current_action = self.STOP_ACTION.copy()
        self.apply_action(self.STOP_ACTION)
        
        if ROBOT_AVAILABLE and self.driver:
            try:
                self.driver.stop()
            except:
                pass
                
        self.get_logger().info("🛑 로봇 정지")

def main(args=None):
    rclpy.init(args=args)
    node = RobotControlNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.stop_robot()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
