# camera_publisher_node.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class CameraPublisherNode(Node):
    def __init__(self):
        super().__init__('camera_publisher_node')
        self.publisher_ = self.create_publisher(Image, '/camera/image_raw', 1)
        self.bridge = CvBridge()

        # Jetson Nano CSI 카메라용 GStreamer 파이프라인
        gst_str = (
            "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, "
            "format=NV12, framerate=30/1 ! nvvidconv ! video/x-raw, format=BGRx ! "
            "videoconvert ! video/x-raw, format=BGR ! appsink"
        )
        self.cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

        if not self.cap.isOpened():
            self.get_logger().error('❌ 카메라 열기 실패 (Jetson GStreamer)')
            rclpy.shutdown()
            return

        self.get_logger().info('✅ 카메라 연결 성공: Enter 키를 누르면 이미지 1회 퍼블리시합니다.')

    def capture_and_publish(self):
        if not self.cap.isOpened():
            self.get_logger().error('🔴 카메라가 열려 있지 않습니다.')
            return False

        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.get_logger().error('⚠️ 프레임 읽기 실패')
            return True  # 카메라는 열려 있지만 프레임을 못 읽은 경우, 재시도 가능

        # 이미지 뒤집기
        frame = cv2.rotate(frame, cv2.ROTATE_180)

        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        self.publisher_.publish(msg)
        self.get_logger().info('📤 이미지 1회 퍼블리시 완료')
        return True

def main(args=None):
    rclpy.init(args=args)
    node = CameraPublisherNode()
    if not rclpy.ok():
        return

    try:
        while rclpy.ok():
            try:
                input("Enter 키를 누르고 한 프레임을 퍼블리시합니다...\n")
            except EOFError:
                break  # 터미널이 닫히는 등 입력이 더 이상 불가능한 경우 종료
            proceed = node.capture_and_publish()
            rclpy.spin_once(node, timeout_sec=0.1)

            if not proceed:
                break  # 카메라가 닫혔거나 심각한 오류가 발생한 경우 루프 탈출

    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info('종료 중: 카메라 및 노드 정리')
        if node.cap.isOpened():
            node.cap.release()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
