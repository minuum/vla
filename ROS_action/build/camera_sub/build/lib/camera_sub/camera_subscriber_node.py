# #camera_publisher_node.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
from datetime import datetime


class CameraSubscriberNode(Node):
    def __init__(self):
        super().__init__('camera_subscriber_node')

        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.listener_callback,
            10
        )
        self.bridge = CvBridge()
        self.frame_count = 0

        # 이미지 저장 경로
        self.save_dir = os.path.join(os.path.expanduser("~"), "camera_images")
        os.makedirs(self.save_dir, exist_ok=True)
        self.get_logger().info(f'📂 저장 경로: {self.save_dir}')

    def listener_callback(self, msg):
        try:
            # 인코딩 정보 출력
            self.get_logger().info(f"📷 수신한 이미지 인코딩: {msg.encoding}")

            # 인코딩 강제하지 않고 메시지의 encoding 그대로 사용
            cv_image = self.bridge.imgmsg_to_cv2(msg, msg.encoding)

            # 필요 시 BGR로 강제 변환 (예: RGB로 저장 시 색 반전되는 경우)
            if msg.encoding == 'rgb8':
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR) 

        except Exception as e:
            self.get_logger().error(f"cv_bridge 변환 실패: {e}")
            return

        # 이미지 저장 
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = os.path.join(self.save_dir, f'image_{timestamp}.jpg')
        cv2.imwrite(filename, cv_image)
        self.frame_count += 1

        self.get_logger().info(f'🖼️ 저장 완료 ({self.frame_count}): {filename}')
				
				#모델에 넘겨줄 때는 cv_image를 넘겨주는 걸로
			
def main(args=None):
    rclpy.init(args=args)
    node = CameraSubscriberNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()