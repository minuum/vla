#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ament_index_python.packages import get_package_share_directory


class CupPublisher(Node):
    def __init__(self):
        super().__init__('cup_publisher')
        self.bridge = CvBridge()
        self.pub = self.create_publisher(Image, '/camera/image_raw', 1)

        # 패키지 공유 디렉토리에서 resource/cup.png 찾기
        pkg_share = get_package_share_directory('object_pose')
        img_path = os.path.join(pkg_share, 'resource', 'cup.png')
        if not os.path.isfile(img_path):
            self.get_logger().error(f'이미지 파일을 찾을 수 없습니다: {img_path}')
            rclpy.shutdown()
            return

        # 이미지 로드
        bgr_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr_img is None:
            self.get_logger().error('cv2.imread 실패')
            rclpy.shutdown()
            return

        # ROS Image 메시지로 변환
        self.msg = self.bridge.cv2_to_imgmsg(bgr_img, encoding='bgr8')

        # 1초 후, 단 한 번만 퍼블리시
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.published = False

    def timer_callback(self):
        if not self.published:
            self.pub.publish(self.msg)
            self.get_logger().info('cup.png 이미지를 /camera/image_raw 토픽으로 퍼블리시했습니다.')
            self.published = True
        else:
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = CupPublisher()
    rclpy.spin(node)


if __name__ == '__main__':
    main()
