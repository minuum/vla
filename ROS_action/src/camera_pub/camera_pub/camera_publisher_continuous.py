#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import time

class MinimalCameraPublisher(Node):
    """최소 카메라 퍼블리셔 - 실제 카메라 또는 가상 카메라"""
    
    def __init__(self):
        super().__init__('minimal_camera_publisher')
        
        # ROS 설정
        self.publisher = self.create_publisher(Image, '/camera/image_raw', 10)
        self.bridge = CvBridge()
        
        # 카메라 초기화
        self.init_camera()
        
        # 타이머 설정 (1Hz)
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.frame_count = 0
        
        self.get_logger().info('📷 최소 카메라 퍼블리셔 시작')
    
    def init_camera(self):
        """카메라 초기화"""
        try:
            # Jetson CSI 카메라 시도
            gst_str = (
                "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=640, height=480, "
                "format=NV12, framerate=30/1 ! nvvidconv ! video/x-raw, format=BGRx ! "
                "videoconvert ! video/x-raw, format=BGR ! appsink drop=true max-buffers=1"
            )
            self.cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
            
            if self.cap.isOpened():
                self.get_logger().info('✅ Jetson CSI 카메라 연결 성공')
                self.camera_type = "실제 카메라"
            else:
                # USB 카메라 시도
                self.cap = cv2.VideoCapture(0)
                if self.cap.isOpened():
                    self.get_logger().info('✅ USB 카메라 연결 성공')
                    self.camera_type = "USB 카메라"
                else:
                    self.get_logger().info('🎨 가상 카메라 모드로 전환')
                    self.cap = None
                    self.camera_type = "가상 카메라"
                    
        except Exception as e:
            self.get_logger().error(f'❌ 카메라 초기화 실패: {e}')
            self.cap = None
            self.camera_type = "가상 카메라"
    
    def get_camera_frame(self):
        """카메라 프레임 가져오기"""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Jetson 카메라는 180도 회전
                if self.camera_type == "실제 카메라":
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                return frame
            else:
                self.get_logger().warn('⚠️ 카메라 프레임 읽기 실패')
                return None
        else:
            return self.generate_virtual_frame()
    
    def generate_virtual_frame(self):
        """가상 카메라 프레임 생성"""
        height, width = 480, 640
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = [50, 100, 50]  # 녹색 배경
        
        # 텍스트 추가
        cv2.putText(frame, f'Virtual Camera Frame {self.frame_count}', 
                   (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # 중앙에 원 그리기
        cv2.circle(frame, (width // 2, height // 2), 100, (0, 0, 255), -1)
        cv2.putText(frame, 'CUP', (width // 2 - 30, height // 2 + 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return frame
    
    def timer_callback(self):
        """타이머 콜백 - 카메라 프레임 발행"""
        try:
            frame = self.get_camera_frame()
            if frame is not None:
                # OpenCV 이미지를 ROS 메시지로 변환
                msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = 'camera_frame'
                
                # 메시지 발행
                self.publisher.publish(msg)
                self.frame_count += 1
                
                self.get_logger().info(f'📷 {self.camera_type} 프레임 발행: {self.frame_count}')
            else:
                self.get_logger().warn('⚠️ 프레임 생성 실패')
                
        except Exception as e:
            self.get_logger().error(f'❌ 타이머 콜백 오류: {e}')
    
    def destroy_node(self):
        """노드 정리"""
        if self.cap and self.cap.isOpened():
            self.cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = MinimalCameraPublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
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
    