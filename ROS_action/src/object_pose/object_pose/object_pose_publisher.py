#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from ros_action_msgs.msg import ObjectInfo
from cv_bridge import CvBridge
import torch
import numpy as np
import cv2

class ObjectPosePublisher(Node):
    def __init__(self):
        super().__init__('object_pose_publisher')

        # CvBridge 초기화
        self.bridge = CvBridge()

        # 카메라 이미지 구독
        self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            1
        )

        # 음성 인식 결과 구독
        self.create_subscription(
            String,
            '/stt/text',
            self.text_callback,
            10
        )

        # ObjectInfo 퍼블리셔
        self.pub = self.create_publisher(ObjectInfo, '/object_info', 1)

        # 탐지할 클래스 (기본값: 'person')
        self.object = 'chair'
        # 첫 추론 여부 플래그
        self.published = False

        # 디바이스 설정
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.get_logger().info(f'Using GPU: {torch.cuda.get_device_name(0)}')
        else:
            self.device = torch.device('cpu')
            self.get_logger().info('CUDA not available, using CPU')

        # YOLOv5 모델 로드
        torch.hub._validate_not_a_forked_repo = lambda a, b=None, c=None: True
        self.model = (
            torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, skip_validation=True)
            .to(self.device)
            .eval()
        )

        # 수평 시야각 (deg)
        self.h_fov =110.0

        self.get_logger().info('ObjectPosePublisher initialized')

    def text_callback(self, msg: String):
        """
        음성 인식 텍스트를 받아서 '사람', '의자', '노트북' 키워드가 포함되면
        self.object를 'person', 'chair', 'laptop'으로 설정
        """
        text = msg.data.strip()
        new_object = None

        if '사람' in text:
            new_object = 'person'
        elif '의자' in text:
            new_object = 'chair'
        elif '노트북' in text:
            new_object = 'laptop'

        if new_object is not None and new_object != self.object:
            self.object = new_object
            # 새로운 클래스에 대해 다시 탐지하도록 published 플래그 리셋
            self.published = False
            self.get_logger().info(f"[STT] Detected keyword, setting object='{self.object}'")

    def convert_to_robot_angle(self, yolo_angle_deg: float) -> float:
        """
        YOLO로 계산된 각도(0° = 정면, 양수 좌측, 음수 우측)를
        로봇 좌표계(0° = 전방, 90° = 좌측, 270° = 우측)로 변환
        """
        return (-yolo_angle_deg) % 360

    def image_callback(self, msg: Image):
        """
        카메라 이미지를 받아 YOLOv5로 객체 탐지 후 ObjectInfo 메시지 퍼블리시
        """
        # 이미 퍼블리시한 클래스가 있으면 무시
        if self.published:
            return

        # 이미지 메시지를 OpenCV 포맷으로 변환
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        h, w = img.shape[:2]

        # BGR -> RGB 변환
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.model(rgb)
        df = results.pandas().xyxy[0]

        # self.object 클래스 필터링
        objects = df[df['name'] == self.object]
        if objects.empty:
            return

        # 가장 큰 바운딩박스를 가진 객체 선택
        objects['area'] = (objects.xmax - objects.xmin) * (objects.ymax - objects.ymin)
        obj = objects.sort_values('area', ascending=False).iloc[0]

        cx = (obj.xmin + obj.xmax) / 2.0       # 바운딩박스 중심 x좌표
        bbox_h = obj.ymax - obj.ymin           # 바운딩박스 높이

        # YOLO 각도 계산 (deg)
        yolo_angle_deg = (cx - w/2) / (w/2) * (self.h_fov / 2)
        robot_angle_deg = self.convert_to_robot_angle(yolo_angle_deg)

        # 거리 계산 (객체 높이 0.15m 가정)
        focal = (h / 2) / np.tan(np.deg2rad(self.h_fov / 2))
        distance = (0.15 * focal) / bbox_h

        # ObjectInfo 메시지 생성 및 퍼블리시
        object_info = ObjectInfo()
        object_info.object_id = self.object
        object_info.distance = float(distance)
        object_info.angle = float(robot_angle_deg)
        self.pub.publish(object_info)

        self.get_logger().info(
            f"Published [ObjectInfo] object={self.object}, "
            f"각도={robot_angle_deg:.1f}°, 거리={distance:.2f}m"
        )
        # 첫 탐지 후 True로 변경
        self.published = True

    def main(self):
        rclpy.spin(self)
        self.destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ObjectPosePublisher()
    try:
        node.main()
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
