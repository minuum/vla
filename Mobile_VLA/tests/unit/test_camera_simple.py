#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import time

class SimpleCameraTest(Node):
    def __init__(self):
        super().__init__('simple_camera_test')
        self.publisher = self.create_publisher(Image, '/camera/image_raw', 10)
        self.bridge = CvBridge()
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.frame_count = 0
        self.get_logger().info('📷 간단한 카메라 테스트 시작')
    
    def timer_callback(self):
        # 간단한 테스트 이미지 생성
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :] = [100, 150, 200]  # 파란색 배경
        
        # 텍스트 추가
        cv2.putText(frame, f'Test Frame {self.frame_count}', 
                   (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        
        # ROS 메시지로 변환 및 발행
        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_frame'
        self.publisher.publish(msg)
        
        self.frame_count += 1
        self.get_logger().info(f'📷 테스트 프레임 발행: {self.frame_count}')

def main(args=None):
    rclpy.init(args=args)
    node = SimpleCameraTest()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except:
            pass
        try:
            rclpy.shutdown()
        except:
            pass

if __name__ == '__main__':
    main()
