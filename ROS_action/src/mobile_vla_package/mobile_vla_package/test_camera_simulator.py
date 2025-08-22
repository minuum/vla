#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np
import time
from std_msgs.msg import String

class TestCameraSimulator(Node):
    """
    테스트용 카메라 시뮬레이터
    색상 변화하는 이미지를 생성하여 카메라 노드 역할
    """
    
    def __init__(self):
        super().__init__('test_camera_simulator')
        
        # 이미지 퍼블리셔
        self.image_pub = self.create_publisher(
            CompressedImage,
            '/camera/image/compressed',
            10
        )
        
        # 제어 서브스크라이버
        self.control_sub = self.create_subscription(
            String,
            '/camera_simulator/control',
            self.control_callback,
            10
        )
        
        # 상태 변수
        self.is_running = True
        self.frame_rate = 10.0  # 10Hz
        self.image_width = 640
        self.image_height = 480
        self.color_shift = 0
        
        # 타이머 설정
        self.timer = self.create_timer(1.0 / self.frame_rate, self.publish_image)
        
        self.get_logger().info("Test Camera Simulator initialized")
    
    def control_callback(self, msg):
        """제어 명령 콜백"""
        try:
            command = msg.data
            if command == 'stop':
                self.is_running = False
                self.get_logger().info("Camera simulator stopped")
            elif command == 'start':
                self.is_running = True
                self.get_logger().info("Camera simulator started")
            elif command.startswith('rate:'):
                new_rate = float(command.split(':')[1])
                self.frame_rate = new_rate
                self.timer.destroy()
                self.timer = self.create_timer(1.0 / self.frame_rate, self.publish_image)
                self.get_logger().info(f"Frame rate changed to {new_rate} Hz")
        except Exception as e:
            self.get_logger().error(f"Error processing control command: {e}")
    
    def generate_test_image(self):
        """테스트 이미지 생성"""
        # 색상 변화하는 이미지 생성
        image = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
        
        # 그라데이션 배경
        for y in range(self.image_height):
            for x in range(self.image_width):
                r = int(128 + 127 * np.sin(x / 50 + self.color_shift))
                g = int(128 + 127 * np.sin(y / 50 + self.color_shift))
                b = int(128 + 127 * np.sin((x + y) / 100 + self.color_shift))
                image[y, x] = [r, g, b]
        
        # 중앙에 원 그리기
        center_x, center_y = self.image_width // 2, self.image_height // 2
        cv2.circle(image, (center_x, center_y), 50, (255, 255, 255), -1)
        
        # 텍스트 추가
        cv2.putText(image, f'Mobile VLA Test {time.time():.1f}', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 색상 변화
        self.color_shift += 0.1
        
        return image
    
    def publish_image(self):
        """이미지 발행"""
        if not self.is_running:
            return
        
        try:
            # 테스트 이미지 생성
            image = self.generate_test_image()
            
            # BGR to RGB 변환
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 압축된 이미지로 변환
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            _, compressed_data = cv2.imencode('.jpg', image_rgb, encode_param)
            
            # CompressedImage 메시지 생성
            msg = CompressedImage()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.format = 'jpeg'
            msg.data = compressed_data.tobytes()
            
            # 이미지 발행
            self.image_pub.publish(msg)
            
        except Exception as e:
            self.get_logger().error(f"Error publishing image: {e}")

def main(args=None):
    rclpy.init(args=args)
    
    node = TestCameraSimulator()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
