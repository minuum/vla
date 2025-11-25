#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

from camera_interfaces.srv import GetImage
from std_srvs.srv import Empty
import threading

from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

class USBCameraServiceServer(Node):
    def __init__(self):
        super().__init__('usb_camera_service_server')
        
        self.bridge = CvBridge()

        self.failed_reads = 0
        self.buffer_lock = threading.Lock()  # 스레드 안전성을 위한 락
        
        # 카메라 초기화
        if not self.init_camera():
            self.get_logger().info('🎨 가상 카메라 모드로 전환 (USB 카메라 시뮬레이션)')

        try:
            self.srv = self.create_service(GetImage, 'get_usb_image_service', self.get_image_callback)
            self.reset_srv = self.create_service(Empty, 'reset_usb_camera_service', self.reset_camera_callback)
            self.get_logger().info('✅ get_usb_image_service 서비스 서버 준비 완료!')
            self.get_logger().info('✅ reset_usb_camera_service 서비스 서버 준비 완료!')
            self.get_logger().info('⏳ USB 카메라 이미지 요청 대기 중...')
        except Exception as e:
            self.get_logger().error(f"❌ USB 카메라 서비스 서버 시작 실패: {e}. 'colcon build' 후 'source install/setup.bash'를 다시 실행했는지, 그리고 패키지 구조가 올바른지 확인하세요.")
            rclpy.shutdown()

    def init_camera(self):
        """USB 카메라를 초기화합니다."""
        # USB 카메라 0번부터 시도
        for camera_id in range(4):  # 0, 1, 2, 3번 카메라 시도
            self.get_logger().info(f'📷 USB 카메라 {camera_id} 시도 중...')
            
            # OpenCV로 USB 카메라 열기
            self.cap = cv2.VideoCapture(camera_id)
            
            if self.cap.isOpened():
                # 카메라 설정
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 버퍼 크기 최소화
                
                # 카메라 웜업: 처음 몇 프레임은 불안정할 수 있으므로 미리 읽어서 버림
                self.get_logger().info(f'🔥 USB 카메라 {camera_id} 웜업 중...')
                for i in range(5):
                    ret, _ = self.cap.read()
                    if not ret:
                        self.get_logger().warn(f'웜업 프레임 {i+1}/5 읽기 실패')
                    else:
                        self.get_logger().info(f'웜업 프레임 {i+1}/5 완료')
                
                self.get_logger().info(f'✅ USB 카메라 {camera_id} 연결 성공!')
                self.failed_reads = 0
                return True
            else:
                self.cap.release()
                self.get_logger().warn(f'⚠️ USB 카메라 {camera_id} 연결 실패')
        
        # 모든 USB 카메라 연결 실패
        self.get_logger().error('❌ 모든 USB 카메라 연결 실패')
        self.cap = None
        return False

    def reset_camera_callback(self, request, response):
        """USB 카메라를 완전히 재시작하여 버퍼를 초기화합니다."""
        with self.buffer_lock:
            self.get_logger().info('🔄 USB 카메라 스트림 완전 재시작 중...')
            
            # 기존 카메라 해제
            if self.cap and self.cap.isOpened():
                self.cap.release()
                self.get_logger().info('📴 기존 USB 카메라 스트림 해제 완료')
            
            # 잠시 대기 (하드웨어 안정화)
            import time
            time.sleep(0.5)
            
            # 카메라 재초기화
            if self.init_camera():
                self.get_logger().info('✅ USB 카메라 스트림 재시작 완료 - 버퍼 초기화됨!')
            else:
                self.get_logger().info('🎨 가상 USB 카메라 모드로 전환')
                
        return response

    def flush_camera_buffer(self):
        """USB 카메라 버퍼에 쌓인 오래된 프레임들을 제거하고 최신 프레임을 가져옵니다."""
        if self.cap is None or not self.cap.isOpened():
            return
            
        with self.buffer_lock:
            self.get_logger().info('🗑️ USB 카메라 버퍼 플러시 시작...')
            
            # 버퍼에 쌓인 프레임들을 빠르게 읽어서 제거 (최대 10개)
            for i in range(10):
                ret, _ = self.cap.read()
                if not ret:
                    break
                    
            self.get_logger().info('✅ USB 카메라 버퍼 플러시 완료')

    def get_fresh_frame(self):
        """버퍼를 플러시하고 최신 프레임을 가져옵니다."""
        if self.cap is None:
            return self.generate_virtual_frame(), "가상 USB 카메라"
            
        with self.buffer_lock:
            # 버퍼 플러시: 빠르게 여러 프레임을 읽어서 버림
            for _ in range(3):
                if self.cap and self.cap.isOpened():
                    self.cap.read()
            
            # 이제 최신 프레임 읽기
            ret, captured_frame = self.cap.read()
            if not ret:
                self.failed_reads += 1
                self.get_logger().warn(f'⚠️ USB 카메라 최신 프레임 읽기 실패 ({self.failed_reads}/5)')
                
                if self.failed_reads >= 5:
                    self.get_logger().error('❌ USB 카메라 하드웨어 문제 감지 - 가상 카메라로 전환')
                    if self.cap.isOpened():
                        self.cap.release()
                    self.cap = None
                    self.failed_reads = 0
                    return self.generate_virtual_frame(), "가상 USB 카메라 (자동 전환)"
                else:
                    return None, "읽기 실패"
            else:
                self.failed_reads = 0
                # USB 카메라는 회전이 필요 없을 수 있음 (필요시 주석 해제)
                # frame = cv2.rotate(captured_frame, cv2.ROTATE_180)
                frame = captured_frame
                return frame, "실제 USB 카메라"

    def get_image_callback(self, request, response):
        frame, camera_type = self.get_fresh_frame()
        
        if frame is not None:
            response.image = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            response.image.header.stamp = self.get_clock().now().to_msg()
            response.image.header.frame_id = 'usb_camera_frame'
            self.get_logger().info(f'📸 {camera_type} 최신 이미지 서비스 요청 처리 완료!')
        else:
            self.get_logger().error('❌ USB 카메라 이미지 캡처/생성 실패 - 서비스 요청에 빈 이미지 반환')
            response.image = Image()

        return response
    
    def generate_virtual_frame(self):
        height, width = 720, 1280
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = [50, 100, 50]
        cv2.putText(frame, f'Mobile VLA Virtual USB Camera', 
                   (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        cv2.circle(frame, (width // 2, height // 2), 50, (0, 0, 255), -1)
        cv2.putText(frame, 'USB', (width // 2 - 30, height // 2 + 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return frame

    def destroy_node(self):
            if hasattr(self, 'cap') and self.cap and self.cap.isOpened():
                self.cap.release()
            super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = USBCameraServiceServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

