# camera_publisher_continuous.py - Mobile VLA용 연속 퍼블리시 카메라 노드
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class CameraPublisherContinuous(Node):
    def __init__(self):
        super().__init__('camera_publisher_continuous')
        self.publisher_ = self.create_publisher(Image, '/camera/image_raw', 10)
        self.bridge = CvBridge()

        # 사용자 제공 정확한 방법: Jetson CSI 카메라용 GStreamer 파이프라인
        gst_str = (
            "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, "
            "format=NV12, framerate=30/1 ! nvvidconv ! video/x-raw, format=BGRx ! "
            "videoconvert ! video/x-raw, format=BGR ! appsink"
        )
        self.cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

        if not self.cap.isOpened():
            self.get_logger().error('❌ 카메라 열기 실패 (Jetson GStreamer)')
            # 가상 카메라 모드로 폴백
            self.cap = None
            self.get_logger().info('🎨 가상 카메라 모드로 전환 (컵 시뮬레이션)')
        else:
            self.get_logger().info('✅ 카메라 연결 성공')

        # 사용자 방식: 3fps 타이머 (모델이 받을 수 있는 정도로 조절)
        timer_period = 1.0 / 3.0  # 초당 3프레임
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.frame_count = 0
        self.failed_reads = 0  # 프레임 읽기 실패 카운터

    def timer_callback(self):
        """사용자 제공 방식: 간단한 타이머 콜백 + 가상 카메라 자동 전환"""
        if self.cap is None:
            # 가상 카메라 모드
            frame = self.generate_virtual_frame()
        else:
            # 실제 카메라에서 프레임 읽기
            ret, frame = self.cap.read()
            if not ret:
                self.failed_reads += 1
                self.get_logger().warn(f'⚠️ 프레임 읽기 실패 ({self.failed_reads}/5)')
                
                # 5번 연속 실패하면 가상 카메라로 전환 (사용자 케이스: 하드웨어 문제)
                if self.failed_reads >= 5:
                    self.get_logger().error('❌ CSI 카메라 하드웨어 문제 감지 - 가상 카메라로 전환')
                    if self.cap.isOpened():
                        self.cap.release()
                    self.cap = None
                    self.failed_reads = 0
                    frame = self.generate_virtual_frame()
                else:
                    return
            else:
                # 성공하면 실패 카운터 리셋
                self.failed_reads = 0
                # 이미지 180도 회전 (사용자 방식: CSI 카메라는 보통 뒤집어져있기 때문)
                frame = cv2.rotate(frame, cv2.ROTATE_180)

        # ROS 이미지 메시지 생성 및 퍼블리시
        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_frame'
        
        self.publisher_.publish(msg)
        self.frame_count += 1
        
        # 9프레임마다 상태 출력 (3fps * 3초 = 9프레임)
        if self.frame_count % 9 == 0:
            camera_type = "가상 카메라" if self.cap is None else "실제 카메라"
            self.get_logger().info(f'📤 {camera_type} 이미지 publish 완료 ({self.frame_count})')
    
    def generate_virtual_frame(self):
        """가상 카메라 프레임 생성 (컵 시뮬레이션)"""
        import numpy as np
        
        # 1280x720 컬러 이미지 생성
        height, width = 720, 1280
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 간단한 패턴 그리기
        frame[:, :] = [50, 100, 50]  # 배경색 (녹색 계열)
        
        # 프레임 카운터 표시
        cv2.putText(frame, f'Mobile VLA Virtual Camera: {self.frame_count}', 
                   (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        # 움직이는 원 (컵 시뮬레이션)
        center_x = int(width // 2 + 200 * np.sin(self.frame_count * 0.1))
        center_y = int(height // 2 + 100 * np.cos(self.frame_count * 0.1))
        cv2.circle(frame, (center_x, center_y), 50, (0, 0, 255), -1)  # 빨간 원
        cv2.putText(frame, 'CUP', (center_x-30, center_y+10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return frame

    def destroy_node(self):
        """사용자 방식: 노드 종료시 카메라 리소스 해제"""
        if hasattr(self, 'cap') and self.cap and self.cap.isOpened():
            self.cap.release()
        super().destroy_node()

def main(args=None):
    """사용자 방식: 간단하고 안정적인 main 함수"""
    rclpy.init(args=args)
    node = CameraPublisherContinuous()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()