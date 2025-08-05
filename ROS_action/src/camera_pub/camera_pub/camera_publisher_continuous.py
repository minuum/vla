#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

from camera_interfaces.srv import GetImage

from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

class CameraServiceServer(Node):
    def __init__(self):
        super().__init__('camera_service_server')
        
        self.bridge = CvBridge()

        gst_str = (
            "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, "
            "format=NV12, framerate=30/1 ! nvvidconv ! video/x-raw, format=BGRx ! "
            "videoconvert ! video/x-raw, format=BGR ! appsink"
        )
        self.cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

        if not self.cap.isOpened():
            self.get_logger().error('❌ 카메라 열기 실패 (Jetson GStreamer)')
            self.cap = None
            self.get_logger().info('🎨 가상 카메라 모드로 전환 (컵 시뮬레이션)')
        else:
            self.get_logger().info('✅ 카메라 연결 성공')

        self.failed_reads = 0

        try:
            self.srv = self.create_service(GetImage, 'get_image_service', self.get_image_callback)
            self.get_logger().info('✅ get_image_service 서비스 서버 준비 완료!')
            self.get_logger().info('⏳ 이미지 요청 대기 중...')
        except Exception as e:
            self.get_logger().error(f"❌ GetImage 서비스 서버 시작 실패: {e}. 'colcon build' 후 'source install/setup.bash'를 다시 실행했는지, 그리고 패키지 구조가 올바른지 확인하세요.")
            rclpy.shutdown()

    def get_image_callback(self, request, response):
        frame = None
        camera_type = "알 수 없음"

        if self.cap is None:
            frame = self.generate_virtual_frame()
            camera_type = "가상 카메라"
        else:
            ret, captured_frame = self.cap.read()
            if not ret:
                self.failed_reads += 1
                self.get_logger().warn(f'⚠️ 프레임 읽기 실패 ({self.failed_reads}/5) - 서비스 요청에 빈 이미지 반환')
                
                if self.failed_reads >= 5:
                    self.get_logger().error('❌ CSI 카메라 하드웨어 문제 감지 - 가상 카메라로 전환')
                    if self.cap.isOpened():
                        self.cap.release()
                    self.cap = None
                    self.failed_reads = 0
                    frame = self.generate_virtual_frame()
                    camera_type = "가상 카메라 (자동 전환)"
                else:
                    response.image = Image()
                    return response
            else:
                self.failed_reads = 0
                frame = cv2.rotate(captured_frame, cv2.ROTATE_180)
                camera_type = "실제 카메라"

            if frame is not None:
                response.image = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
                response.image.header.stamp = self.get_clock().now().to_msg()
                response.image.header.frame_id = 'camera_frame'
                self.get_logger().info(f'📸 {camera_type} 이미지 서비스 요청 처리 완료!')
            else:
                self.get_logger().error('❌ 이미지 캡처/생성 실패 - 서비스 요청에 빈 이미지 반환')
                response.image = Image()

            return response
        
        def generate_virtual_frame(self):
            height, width = 720, 1280
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :] = [50, 100, 50]
            cv2.putText(frame, f'Mobile VLA Virtual Camera', 
                       (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            cv2.circle(frame, (width // 2, height // 2), 50, (0, 0, 255), -1)
            cv2.putText(frame, 'CUP', (width // 2 - 30, height // 2 + 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return frame

        def destroy_node(self):
            if hasattr(self, 'cap') and self.cap and self.cap.isOpened():
                self.cap.release()
            super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = CameraServiceServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
    