#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import Twist
import torch
#from my_vla_model import VLAModel
from robovlms.model.backbone import VLAModel
from cv_bridge import CvBridge

class VLAInferenceNode(Node):
    def __init__(self):
        super().__init__('vla_node')
        self.model = VLAModel.from_pretrained('path/to/model')  # 모델 경로
        self.model.eval()
        self.bridge = CvBridge()

        self.image = None
        self.text = None

        # 입력 구독
        self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.create_subscription(String, '/stt/text', self.text_callback, 10)

        # 두 가지 출력 방식
        self.selected_action_pub = self.create_publisher(String, '/selected_action', 10)  # 방식 1: 선택된 확률이 가장 높은 명령
        # self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)  # 방식 2: 연속 벡터

    def image_callback(self, msg):
        self.image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.try_inference()

    def text_callback(self, msg):
        self.text = msg.data
        self.try_inference()
        
    def try_inference(self):
        if self.image is not None and self.text is not None:
            # 방식 1: 확률이 가장 높은 명령 선택 (문자열)
            selected_action = self.model.predict_action(self.image, self.text)
            self.get_logger().info(f"모델이 선택한 확률이 가장 높은 동작: {selected_action}")
            self.selected_action_pub.publish(String(data=selected_action))

            # 방식 2: 연속 벡터
            # action_vec = self.model.predict(self.image, self.text)  # 예: [0.2, 0.1, -0.3]
            # twist = self.convert_to_twist(action_vec)
            # self.cmd_vel_pub.publish(twist)
            # self.get_logger().info(f"[연속벡터 출력] 예측 벡터: {action_vec}")

            # 입력 초기화
            self.image = None
            self.text = None

    # 방식2에서 벡터인 action을 twist 메시지로 변환
    #def convert_to_twist(self, action_vec):
    #    msg = Twist()
    #    msg.linear.x = action_vec[0]
    #    msg.linear.y = action_vec[1]
    #    msg.angular.z = action_vec[2]
    #    return msg

def main(args=None):
    rclpy.init(args=args)
    node = VLAInferenceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
