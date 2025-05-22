#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from pop.driving import Driving
from pop.Psd import Psd
from pop.Ultrasonic import Ultrasonic

# linear x 0.5, y 0.5시에 좌측 앞 이동
# linear x 0.5, y -0.5시에 우측 앞 이동
# linear x 0.5, y 0.0시에 정면 이동
# linear x 0.0, y 0.5시에 좌측 이동
# linear x 0.0, y -0.5시에 우측 이동
class OmniDriveNode(Node):
    def __init__(self):
        super().__init__('omni_drive_node')
        self.get_logger().info('OmniDriveNode 초기화 완료')

        # Driving 모듈
        try:
            self.car = Driving()
            self.get_logger().info('Driving 모듈 연결 성공')
        except Exception as e:
            self.get_logger().error(f'Driving 초기화 실패: {e}')

        # 센서
        self.psd = Psd(dev="can0", bitrate=500000)
        self.us  = Ultrasonic(dev="can0", bitrate=500000)

        # 최근 cmd_vel 저장
        self.lx = 0.0
        self.ly = 0.0

        # 구독
        self.create_subscription(Twist, 'cmd_vel',
                                 self.cmd_vel_callback, 10)

        # 제어 루프 타이머 (10Hz)
        self.create_timer(0.1, self.control_loop)

        # 스로틀 스케일
        MAX_LINEAR_VEL = 1.0
        self.LINEAR_SCALE = 100.0 / MAX_LINEAR_VEL

    def cmd_vel_callback(self, msg: Twist):
        self.lx = msg.linear.x
        self.ly = msg.linear.y
        self.get_logger().debug(f'cmd_vel 수신: x={self.lx:.2f}, y={self.ly:.2f}')

    def control_loop(self):
        # 센서 읽기 (연속)
        psd_min = min(self.psd.read())
        us_min  = min(self.us.read())

        if psd_min <= 20 or us_min <= 20:
            # 장애물 가까이 → 정지
            self.get_logger().info(f'장애물 감지(PSD={psd_min}, US={us_min}) → 정지')
            self.car.stop()
            return

        # 선형 유효성 검사(0이면 stop)
        if abs(self.lx) < 1e-3 and abs(self.ly) < 1e-3:
            self.car.stop()
            return

        # 이동
        angle_deg    = math.degrees(math.atan2(self.ly, self.lx))
        throttle_lin = max(1, round(math.hypot(self.lx, self.ly) * self.LINEAR_SCALE))
        self.get_logger().info(f'이동: angle={angle_deg:.1f}°, throttle={throttle_lin}')
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
