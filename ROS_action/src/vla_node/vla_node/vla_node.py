#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from audio_common_msgs.msg import AudioData # Main Flow에 따름
# from geometry_msgs.msg import Twist # 필요시 주석 해제

import cv2
from cv_bridge import CvBridge
import torch
from PIL import Image as PILImage
# import numpy as np # 필요시 사용
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
# from transformers.utils import logging as transformers_logging # 상세 로깅 제어시

# big_vision 및 기타 외부 라이브러리 임포트는 ROS 2 패키지 환경에 맞게 처리 필요
# 예를 들어, 해당 라이브러리들을 이 패키지의 의존성으로 추가하거나
# 또는 시스템 전체에 설치되어 Python이 찾을 수 있도록 해야 함.
# 지금은 ROS 2 노드 핵심 로직에 집중.

class VlaNode(Node):
    def __init__(self):
        super().__init__('vla_node')
        self.get_logger().info(f"***** Initializing {self.get_name()} *****")

        # ROS 파라미터 선언
        self.declare_parameter('vla_model_id', 'google/paligemma-3b-mix-224')
        # self.declare_parameter('vla_model_revision', 'global_step-1010000') # 특정 리비전 필요시
        self.declare_parameter('vla_model_revision', 'main') # 기본값 'main' 사용
        self.declare_parameter('torch_dtype_str', 'float16')  # 'float16', 'bfloat16', 'float32'
        self.declare_parameter('low_cpu_mem_usage', True)
        self.declare_parameter('device', 'cuda')  # 'cuda' or 'cpu'
        self.declare_parameter('max_new_tokens', 128)
        # self.declare_parameter('hf_token', '') # 비공개 모델 접근 시 Hugging Face 토큰

        # 파라미터 값 가져오기
        vla_model_id = self.get_parameter('vla_model_id').get_parameter_value().string_value
        vla_model_revision = self.get_parameter('vla_model_revision').get_parameter_value().string_value
        torch_dtype_str = self.get_parameter('torch_dtype_str').get_parameter_value().string_value
        low_cpu_mem_usage = self.get_parameter('low_cpu_mem_usage').get_parameter_value().bool_value
        self.device = self.get_parameter('device').get_parameter_value().string_value
        self.max_new_tokens = self.get_parameter('max_new_tokens').get_parameter_value().integer_value
        # hf_token = self.get_parameter('hf_token').get_parameter_value().string_value

        # Torch Dtype 설정
        if torch_dtype_str == 'float16': self.torch_dtype = torch.float16
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
