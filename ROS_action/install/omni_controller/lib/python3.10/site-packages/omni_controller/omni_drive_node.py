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

        # Driving 모듈 초기화 및 PID 활성화
        try:
            self.car = Driving()
            self.car.pid = True
            self.get_logger().info('Driving 모듈 연결 성공 (PID ON)')
        except Exception as e:
            self.get_logger().error(f'Driving 모듈 초기화 실패: {e}')

        # 센서 초기화
        self.psd = Psd(dev="can0", bitrate=500000)
        self.us  = Ultrasonic(dev="can0", bitrate=500000)

        # 명령 저장 변수
        self.lx = 0.0
        self.ly = 0.0

        # 휠 배치 각도 (라디안): 0°, 120°, 240°
        self.wheel_angles = [
            0,
            2 * math.pi / 3,
            4 * math.pi / 3,
        ]

        # 스로틀 스케일
        MAX_LINEAR_VEL = 1.0  # m/s
        self.LINEAR_SCALE = 100.0 / MAX_LINEAR_VEL

        # 구독 및 타이머
        self.create_subscription(Twist, 'cmd_vel',
                                 self.cmd_vel_callback, 10)
        self.create_timer(0.1, self.control_loop)

    def cmd_vel_callback(self, msg: Twist):
        self.lx = msg.linear.x
        self.ly = msg.linear.y

    def control_loop(self):
        # 센서 값 읽기
        psd_min = min(self.psd.read())
        us_min  = min(self.us.read())
        if psd_min <= 20 or us_min <= 20:
            self.get_logger().info(f'장애물 감지 (PSD={psd_min}, US={us_min}) → 정지')
            self.car.stop()
            return

        # 유효한 선형 입력 검사
        if abs(self.lx) < 1e-3 and abs(self.ly) < 1e-3:
            self.car.stop()
            return

        # 순수 병진(직선) 벡터 계산
        angle_rad    = math.atan2(self.ly, self.lx)
        throttle_lin = max(1, round(math.hypot(self.lx, self.ly) * self.LINEAR_SCALE))
        self.get_logger().info(
            f'직선 이동: 각도={math.degrees(angle_rad):.1f}°, 스로틀={throttle_lin}'
        )

        # 각 휠 속도 설정 (회전 성분 제거)
        for i, theta in enumerate(self.wheel_angles):
            speed = throttle_lin * math.cos(angle_rad - theta)
            self.car.wheel_vec[i] = self.car.WHEEL_CENTER + round(speed)

        # 휠 속도 전송
        self.car.transfer()

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
