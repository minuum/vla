#!/usr/bin/env python3
"""
최소 기능 ROS2 노드들
카메라, 추론, 제어, 모니터링 노드
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import time
import threading

class MinimalCameraNode(Node):
    """최소 카메라 노드"""
    
    def __init__(self):
        super().__init__('minimal_camera_node')
        self.publisher = self.create_publisher(String, '/camera/image_raw', 10)
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.counter = 0
        self.get_logger().info('📷 최소 카메라 노드 시작')
    
    def timer_callback(self):
        msg = String()
        msg.data = f'Camera Frame {self.counter}'
        self.publisher.publish(msg)
        self.counter += 1
        self.get_logger().info(f'📷 카메라 프레임 발행: {msg.data}')

class MinimalInferenceNode(Node):
    """최소 추론 노드"""
    
    def __init__(self):
        super().__init__('minimal_inference_node')
        self.subscription = self.create_subscription(
            String, '/camera/image_raw', self.camera_callback, 10)
        self.publisher = self.create_publisher(String, '/inference/result', 10)
        self.get_logger().info('🧠 최소 추론 노드 시작')
    
    def camera_callback(self, msg):
        # 간단한 추론 시뮬레이션
        result_msg = String()
        result_msg.data = f'Inference Result for {msg.data} - Action: [0.5, 0.3]'
        self.publisher.publish(result_msg)
        self.get_logger().info(f'🧠 추론 결과 발행: {result_msg.data}')

class MinimalControlNode(Node):
    """최소 제어 노드"""
    
    def __init__(self):
        super().__init__('minimal_control_node')
        self.subscription = self.create_subscription(
            String, '/inference/result', self.inference_callback, 10)
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.get_logger().info('🤖 최소 제어 노드 시작')
    
    def inference_callback(self, msg):
        # 간단한 제어 명령 생성
        cmd_msg = Twist()
        cmd_msg.linear.x = 0.5
        cmd_msg.angular.z = 0.1
        self.publisher.publish(cmd_msg)
        self.get_logger().info(f'🤖 제어 명령 발행: linear.x={cmd_msg.linear.x}, angular.z={cmd_msg.angular.z}')

class MinimalMonitorNode(Node):
    """최소 모니터링 노드"""
    
    def __init__(self):
        super().__init__('minimal_monitor_node')
        self.camera_sub = self.create_subscription(
            String, '/camera/image_raw', self.camera_callback, 10)
        self.inference_sub = self.create_subscription(
            String, '/inference/result', self.inference_callback, 10)
        self.control_sub = self.create_subscription(
            Twist, '/cmd_vel', self.control_callback, 10)
        self.get_logger().info('📊 최소 모니터링 노드 시작')
    
    def camera_callback(self, msg):
        self.get_logger().info(f'📊 카메라 모니터링: {msg.data}')
    
    def inference_callback(self, msg):
        self.get_logger().info(f'📊 추론 모니터링: {msg.data}')
    
    def control_callback(self, msg):
        self.get_logger().info(f'📊 제어 모니터링: linear.x={msg.linear.x}, angular.z={msg.angular.z}')

def main_camera():
    """카메라 노드 메인"""
    rclpy.init()
    node = MinimalCameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

def main_inference():
    """추론 노드 메인"""
    rclpy.init()
    node = MinimalInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

def main_control():
    """제어 노드 메인"""
    rclpy.init()
    node = MinimalControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

def main_monitor():
    """모니터링 노드 메인"""
    rclpy.init()
    node = MinimalMonitorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        node_type = sys.argv[1]
        if node_type == 'camera':
            main_camera()
        elif node_type == 'inference':
            main_inference()
        elif node_type == 'control':
            main_control()
        elif node_type == 'monitor':
            main_monitor()
        else:
            print("사용법: python3 minimal_ros2_nodes.py [camera|inference|control|monitor]")
    else:
        print("사용법: python3 minimal_ros2_nodes.py [camera|inference|control|monitor]")
