#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np
import time
from std_msgs.msg import String
import threading

class TestCameraSimulator(Node):
    """
    🎨 향상된 테스트용 카메라 시뮬레이터
    camera_publisher_continuous.py 기반으로 개선된 버전
    """
    
    def __init__(self):
        super().__init__('test_camera_simulator')
        
        # 이미지 퍼블리셔 (QoS 설정 추가)
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=1
        )
        
        self.image_pub = self.create_publisher(
            CompressedImage,
            '/camera/image/compressed',
            qos
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
        self.frame_rate = 30.0  # 30Hz로 증가
        self.image_width = 1280  # 고해상도로 증가
        self.image_height = 720
        self.color_shift = 0
        self.frame_count = 0
        self.last_log_time = time.time()
        
        # 스레드 안전성을 위한 락
        self.buffer_lock = threading.Lock()
        
        # 타이머 설정
        self.timer = self.create_timer(1.0 / self.frame_rate, self.publish_image)
        
        self.get_logger().info("🎨 Enhanced Test Camera Simulator initialized")
        self.get_logger().info(f"📸 Resolution: {self.image_width}x{self.image_height}")
        self.get_logger().info(f"⚡ Frame Rate: {self.frame_rate} Hz")
        self.get_logger().info(f"📡 Topic: /camera/image/compressed")
    
    def control_callback(self, msg):
        """제어 명령 콜백"""
        try:
            command = msg.data
            if command == 'stop':
                self.is_running = False
                self.get_logger().info("🛑 Camera simulator stopped")
            elif command == 'start':
                self.is_running = True
                self.get_logger().info("▶️ Camera simulator started")
            elif command.startswith('rate:'):
                new_rate = float(command.split(':')[1])
                self.frame_rate = new_rate
                self.timer.destroy()
                self.timer = self.create_timer(1.0 / self.frame_rate, self.publish_image)
                self.get_logger().info(f"⚡ Frame rate changed to {new_rate} Hz")
            elif command == 'status':
                self.get_logger().info(f"📊 Status: Running={self.is_running}, FPS={self.frame_rate}, Frames={self.frame_count}")
        except Exception as e:
            self.get_logger().error(f"❌ Error processing control command: {e}")
    
    def generate_test_image(self):
        """🎨 향상된 테스트 이미지 생성 (컵 시뮬레이션)"""
        with self.buffer_lock:
            # 고해상도 이미지 생성
            image = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
            
            # 배경 그라데이션 (더 부드럽게)
            for y in range(self.image_height):
                for x in range(self.image_width):
                    r = int(50 + 50 * np.sin(x / 100 + self.color_shift))
                    g = int(100 + 50 * np.sin(y / 100 + self.color_shift))
                    b = int(50 + 50 * np.sin((x + y) / 200 + self.color_shift))
                    image[y, x] = [r, g, b]
            
            # 🏆 컵 시뮬레이션 (Mobile VLA용)
            center_x, center_y = self.image_width // 2, self.image_height // 2
            
            # 컵 본체 (원형)
            cv2.circle(image, (center_x, center_y), 80, (200, 150, 100), -1)
            cv2.circle(image, (center_x, center_y), 80, (100, 50, 0), 3)
            
            # 컵 안쪽 (음료)
            cv2.circle(image, (center_x, center_y), 60, (0, 100, 200), -1)
            
            # 컵 손잡이
            handle_x = center_x + 100
            cv2.ellipse(image, (handle_x, center_y), (20, 40), 0, 0, 180, (150, 100, 50), 5)
            
            # 텍스트 추가
            cv2.putText(image, f'Mobile VLA Cup Simulation', 
                       (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            cv2.putText(image, f'Frame: {self.frame_count} | Time: {time.time():.1f}', 
                       (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(image, f'Target: CUP', 
                       (center_x - 60, center_y + 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # 색상 변화
            self.color_shift += 0.05
            
            return image
    
    def publish_image(self):
        """📸 이미지 발행 (개선된 버전)"""
        if not self.is_running:
            return
        
        try:
            # 테스트 이미지 생성
            image = self.generate_test_image()
            
            # BGR to RGB 변환
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 압축된 이미지로 변환 (고품질)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
            _, compressed_data = cv2.imencode('.jpg', image_rgb, encode_param)
            
            # CompressedImage 메시지 생성
            msg = CompressedImage()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'camera_frame'
            msg.format = 'jpeg'
            msg.data = compressed_data.tobytes()
            
            # 이미지 발행
            self.image_pub.publish(msg)
            
            # 프레임 카운터 증가
            self.frame_count += 1
            
            # 주기적 로깅 (1초마다)
            current_time = time.time()
            if current_time - self.last_log_time >= 1.0:
                fps = self.frame_count / (current_time - self.last_log_time + 1.0)
                self.get_logger().info(f"📸 Published frame {self.frame_count} | FPS: {fps:.1f} | Size: {len(compressed_data)} bytes")
                self.frame_count = 0
                self.last_log_time = current_time
            
        except Exception as e:
            self.get_logger().error(f"❌ Error publishing image: {e}")

def main(args=None):
    rclpy.init(args=args)
    
    node = TestCameraSimulator()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("🛑 Camera simulator interrupted by user")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
