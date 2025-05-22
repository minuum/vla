#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from pop.driving import Driving
from pop.Psd import Psd
from pop.Ultrasonic import Ultrasonic

class OmniDriveNode(Node):
    def __init__(self):
        super().__init__('omni_drive_node')
        self.get_logger().info('OmniDriveNode 초기화 완료')

        # Driving 모듈 초기화
        try:
            self.car = Driving()
            self.get_logger().info('Driving 모듈 연결 성공')
        except Exception as e:
            self.get_logger().error(f'Driving 모듈 초기화 실패: {e}')

        # PSD·Ultrasonic 센서 초기화
        self.psd = Psd(dev="can0", bitrate=500000)
        self.us  = Ultrasonic(dev="can0", bitrate=500000)

        # 최근 cmd_vel 저장
        self.lx = 0.0
        self.ly = 0.0

        # cmd_vel 구독
        self.create_subscription(Twist, 'cmd_vel',
                                 self.cmd_vel_callback, 10)

        # 제어 루프 타이머 (10Hz)
        self.create_timer(0.1, self.control_loop)

        # 스로틀 스케일
        MAX_LINEAR_VEL = 1.0  # m/s
        self.LINEAR_SCALE = 100.0 / MAX_LINEAR_VEL

    def cmd_vel_callback(self, msg: Twist):
        self.lx = msg.linear.x
        self.ly = msg.linear.y

    def control_loop(self):
        # 1) 센서 읽기
        psd_min = min(self.psd.read())
        us_min  = min(self.us.read())
        if psd_min <= 20 or us_min <= 20:
            self.get_logger().info(f'장애물 감지 (PSD={psd_min}, US={us_min}) → 정지')
            self.car.stop()
            return

        # 2) 선형 입력 유효성 검사
        if abs(self.lx) < 1e-3 and abs(self.ly) < 1e-3:
            self.car.stop()
            return

        # 3) 순수 병진(직선) 주행
        angle_rad    = math.atan2(self.ly, self.lx)
        angle_deg    = math.degrees(angle_rad)
        throttle_lin = max(1, round(math.hypot(self.lx, self.ly) * self.LINEAR_SCALE))

        self.get_logger().info(f'직선 이동: 각도={angle_deg:.1f}°, 스로틀={throttle_lin}')
        # Driving.move() 는 순수 병진만 수행
        self.car.move(angle_deg, throttle_lin)

def main(args=None):
    rclpy.init(args=args)
    node = OmniDriveNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 