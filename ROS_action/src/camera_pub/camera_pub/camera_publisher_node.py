#camera_publisher_node.py
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
            return

        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error('⚠️ 프레임 읽기 실패')
        else:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            self.publisher_.publish(msg)
            self.get_logger().info('📤 이미지 1회 publish 완료')

def main(args=None):
    rclpy.init(args=args)
    node = CameraPublisherNode()
    rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
