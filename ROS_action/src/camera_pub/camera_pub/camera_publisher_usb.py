# camera_publisher_usb.py - USB 카메라 전용 연속 퍼블리시 노드
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class CameraPublisherUSB(Node):
    def __init__(self):
        super().__init__('camera_publisher_usb')
        self.publisher_ = self.create_publisher(Image, '/camera/image_raw', 10)
        self.bridge = CvBridge()

        # USB 카메라만 시도 (CSI 카메라 건너뜀)
        camera_devices = [0, 1, 2]  # 여러 USB 포트 시도
        self.cap = None
        
        for device_id in camera_devices:
            self.get_logger().info(f'📷 USB 카메라 {device_id} 시도 중...')
            cap = cv2.VideoCapture(device_id)
            
            if cap.isOpened():
                # 테스트 프레임 읽기
                ret, frame = cap.read()
                if ret and frame is not None:
                    self.cap = cap
                    self.get_logger().info(f'✅ USB 카메라 {device_id} 연결 성공!')
                    break
                else:
                    cap.release()
            else:
                cap.release()
                
        if self.cap is None:
            self.get_logger().error('❌ 사용 가능한 USB 카메라 없음')
            
            # 가상 카메라 생성 (테스트용)
            self.get_logger().info('🎨 가상 카메라 모드로 전환')
            self.use_virtual_camera = True
        else:
            self.use_virtual_camera = False
            
            # 카메라 설정 최적화
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # 연속 퍼블리시 타이머 설정 (10Hz)
        self.timer = self.create_timer(0.1, self.capture_and_publish)
        
        self.frame_count = 0
        self.get_logger().info('🎥 USB 카메라 연속 퍼블리시 시작 (10Hz)')

    def generate_virtual_frame(self):
        """가상 카메라 프레임 생성 (테스트용)"""
        import numpy as np
        
        # 640x480 컬러 이미지 생성
        height, width = 480, 640
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 간단한 패턴 그리기
        frame[:, :] = [50, 50, 100]  # 배경색
        
        # 프레임 카운터 표시
        cv2.putText(frame, f'Virtual Camera Frame: {self.frame_count}', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 움직이는 사각형
        x = (self.frame_count * 2) % (width - 100)
        y = (self.frame_count * 1) % (height - 100)
        cv2.rectangle(frame, (x, y), (x+100, y+100), (0, 255, 0), -1)
        
        return frame

    def capture_and_publish(self):
        frame = None
        
        if self.use_virtual_camera:
            # 가상 카메라 프레임 생성
            frame = self.generate_virtual_frame()
            
        else:
            # 실제 USB 카메라에서 프레임 읽기
            if not self.cap.isOpened():
                self.get_logger().error('🔴 USB 카메라가 열려 있지 않습니다.')
                return

            ret, frame = self.cap.read()
            if not ret or frame is None:
                self.get_logger().warn('⚠️ USB 카메라 프레임 읽기 실패')
                return

        # ROS 이미지 메시지 생성 및 퍼블리시
        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'usb_camera_frame'
        
        self.publisher_.publish(msg)
        
        self.frame_count += 1
        
        # 10초마다 상태 출력
        if self.frame_count % 100 == 0:  # 10Hz * 10초 = 100프레임
            camera_type = "가상 카메라" if self.use_virtual_camera else "USB 카메라"
            self.get_logger().info(f'📸 {camera_type} 프레임 {self.frame_count}개 퍼블리시 완료')

    def __del__(self):
        if hasattr(self, 'cap') and self.cap and self.cap.isOpened():
            self.cap.release()

def main(args=None):
    rclpy.init(args=args)
    node = CameraPublisherUSB()
    
    if not rclpy.ok():
        return

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info('🛑 종료 중: USB 카메라 및 노드 정리')
        if hasattr(node, 'cap') and node.cap and node.cap.isOpened():
            node.cap.release()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()