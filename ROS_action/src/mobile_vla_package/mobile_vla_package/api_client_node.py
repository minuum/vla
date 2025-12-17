#!/usr/bin/env python3
"""
Mobile VLA 추론 클라이언트 (Jetson용)
원격 API 서버와 통신
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
import cv2
import numpy as np
import requests
import base64
import json
from collections import deque

class MobileVLAAPIClient(Node):
    def __init__(self, api_server_url="http://192.168.1.100:8000"):
        super().__init__('mobile_vla_api_client')
        
        self.api_server_url = api_server_url
        self.image_buffer = deque(maxlen=2)  # 최신 이미지만 사용
        
        # ROS 설정
        self.image_sub = self.create_subscription(
            CompressedImage,
            '/camera/image/compressed',
            self.image_callback,
            10
        )
        
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )
        
        # 타이머 (100ms마다 추론 - Reactive control)
        self.timer = self.create_timer(0.1, self.inference_timer_callback)
        
        self.get_logger().info(f"✅ API 클라이언트 시작 (서버: {api_server_url})")
    
    def image_callback(self, msg):
        """카메라 이미지 수신"""
        # 압축 이미지 → numpy
        np_arr = np.frombuffer(msg.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        # Resize to 224x224
        image = cv2.resize(image, (224, 224))
        
        # 버퍼에 추가
        self.image_buffer.append(image)
    
    def inference_timer_callback(self):
        """추론 타이머 (100ms - Reactive control)"""
        if len(self.image_buffer) < 1:  # 최소 1개 이미지만 필요
            return
        
        try:
            # 최신 이미지 하나만 사용
            latest_img = self.image_buffer[-1]
            
            # Base64 인코딩
            _, buffer = cv2.imencode('.jpg', latest_img)
            img_b64 = base64.b64encode(buffer).decode('utf-8')
            
            # API 요청 (단일 이미지)
            response = requests.post(
                f"{self.api_server_url}/predict",
                json={
                    "image": img_b64,  # 단일 이미지
                    "instruction": "Navigate around obstacles and reach the front of the beverage bottle on the left"
                },
                timeout=1.0  # 1초 타임아웃
            )
            
            if response.status_code == 200:
                data = response.json()
                action = data["action"]  # [linear_x, linear_y]
                
                # 액션 실행 (2 DOF)
                twist = Twist()
                twist.linear.x = float(action[0])   # linear_x
                twist.linear.y = float(action[1])   # linear_y
                twist.angular.z = 0.0               # 우리 태스크에서는 사용 안 함
                self.cmd_vel_pub.publish(twist)
                
                self.get_logger().info(
                    f"✅ 추론: {data['latency_ms']:.1f}ms, "
                    f"Action: [x={action[0]:.3f}, y={action[1]:.3f}]"
                )
            else:
                self.get_logger().error(f"❌ API 에러: {response.status_code}")
        
        except requests.Timeout:
            self.get_logger().warn("⏱️ API 타임아웃")
        except Exception as e:
            self.get_logger().error(f"❌ 추론 실패: {e}")

def main(args=None):
    rclpy.init(args=args)
    
    # API 서버 URL (환경변수 또는 인자로 설정)
    import os
    api_url = os.getenv("VLA_API_SERVER", "http://localhost:8000")
    
    node = MobileVLAAPIClient(api_server_url=api_url)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
