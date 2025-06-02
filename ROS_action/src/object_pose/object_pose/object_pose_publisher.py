#object_pose_publisher.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from ros_action_msgs.msg import ObjectInfo
from cv_bridge import CvBridge
import torch
import numpy as np
import cv2

class ObjectPosePublisher(Node):
    def __init__(self):
        super().__init__('object_pose_publisher')
        self.bridge = CvBridge()
        self.create_subscription(Image, '/camera/image_raw', self.callback, 1)
        self.pub = self.create_publisher(ObjectInfo, '/object_info', 1)

        # 탐지할 클래스 변수
        self.object = 'person'
        self.published = False  # 첫 추론 여부 플래그

        # 디바이스스 설정
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
        self.h_fov = 120.0  # 수평 시야각(deg)

    def convert_to_robot_angle(self, yolo_angle_deg):
        """YOLO 각도 → 로봇 좌표계 변환 (0°=전방, 90°=좌측, 270°=우측)"""
        return (-yolo_angle_deg) % 360

    def callback(self, msg: Image):
        if self.published:
            return  # 이미 퍼블리시 했으면 무시

        # 이미지 메시지를 OpenCV 포맷으로 변환
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        h, w = img.shape[:2]

        # YOLOv5에 입력할 RGB 이미지 생성
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.model(rgb)
        df = results.pandas().xyxy[0]

        # self.object 클래스만 필터링
        objects = df[df['name'] == self.object]
        if objects.empty:
            return

        # 가장 큰 바운딩박스를 가진 객체 선택
        objects['area'] = (objects.xmax - objects.xmin) * (objects.ymax - objects.ymin)
        obj = objects.sort_values('area', ascending=False).iloc[0]

        cx = (obj.xmin + obj.xmax) / 2.0       # 바운딩박스 중심 x좌표
        bbox_h = obj.ymax - obj.ymin           # 바운딩박스 높이

        # YOLO 각도 계산 (deg)
        yolo_angle_deg = (cx - w/2) / (w/2) * (self.h_fov/2)
        robot_angle_deg = self.convert_to_robot_angle(yolo_angle_deg)

        # 거리 계산 (객체 높이 0.15m 가정, 추후 확장 가능)
        focal = (h/2) / np.tan(np.deg2rad(self.h_fov/2))
        distance = (0.15 * focal) / bbox_h

        # ObjectInfo 메시지 생성 및 퍼블리시
        object_info = ObjectInfo()
        object_info.object_id = self.object
        object_info.distance = float(distance)
        object_info.angle = float(robot_angle_deg)
        self.pub.publish(object_info)

        self.get_logger().info(
            f'Published [ObjectInfo] object={self.object}, 각도={robot_angle_deg:.1f}°, 거리={distance:.2f}m'
        )
        self.published = True  # 첫 추론 후 True로 변경

def main(args=None):
    rclpy.init(args=args)
    node = ObjectPosePublisher()
    rclpy.spin(node)  
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
