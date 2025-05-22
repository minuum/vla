# omni_drive_node.py
import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from pop.driving import Driving

class OmniDriveNode(Node):
    def __init__(self):
        super().__init__('omni_drive_node')
        self.get_logger().info('OmniDriveNode 초기화 완료, cmd_vel 구독 대기 중…')

        try:
            self.car = Driving()
            self.get_logger().info('Driving 모듈 연결 성공')
        except Exception as e:
            self.get_logger().error(f'Driving 모듈 초기화 실패: {e}')

        self.subscription = self.create_subscription(
            Twist,
            'cmd_vel',
            self.cmd_vel_callback,
            10
        )

        # m/s 또는 rad/s → 0~100 스로틀 매핑
        MAX_LINEAR_VEL  = 1.0  # m/s
        MAX_ANGULAR_VEL = 1.0  # rad/s
        self.LINEAR_SCALE  = 100.0 / MAX_LINEAR_VEL
        self.ANGULAR_SCALE = 100.0 / MAX_ANGULAR_VEL

    def cmd_vel_callback(self, msg: Twist):
        lx, ly, az = msg.linear.x, msg.linear.y, msg.angular.z
        self.get_logger().info(f'[cmd_vel] lin=({lx:.2f},{ly:.2f}), ang={az:.2f}')

        eps = 1e-3
        linear_valid  = abs(lx) > eps or abs(ly) > eps
        angular_valid = abs(az) > eps

        # 1) 곡선 주행 (translation + rotation)
        if linear_valid and angular_valid:
            angle_rad     = math.atan2(ly, lx)
            throttle_lin  = max(1, round(math.hypot(lx, ly) * self.LINEAR_SCALE))
            throttle_ang  = round(az * self.ANGULAR_SCALE)
            self.get_logger().info(f'→ 곡선 주행: lin_thr={throttle_lin}, ang_thr={throttle_ang}')

            # 각 바퀴별 속도 계산 (3-Omniwheel: 0°, 120°, 240°)
            wheel_angles = [0, 2 * math.pi / 3, 4 * math.pi / 3]
            for i, theta in enumerate(wheel_angles):
                # 로컬 x,y 성분과 회전 성분 합산
                speed = throttle_lin * math.cos(angle_rad - theta) + throttle_ang
                self.car.wheel_vec[i] = self.car.WHEEL_CENTER + round(speed)

            # 한번에 전송
            self.car.transfer()
            return

        # 2) 순수 평면 이동
        if linear_valid:
            angle_deg    = math.degrees(math.atan2(ly, lx))
            throttle_lin = max(1, round(math.hypot(lx, ly) * self.LINEAR_SCALE))
            self.get_logger().info(f'→ 이동: angle={angle_deg:.1f}°, throttle={throttle_lin}')
            self.car.move(angle_deg, throttle_lin)
            return

        # 3) 순수 회전
        if angular_valid:
            throttle_ang = round(az * self.ANGULAR_SCALE)
            self.get_logger().info(f'→ 회전: throttle={throttle_ang}')
            self.car.spin(throttle_ang)
            return

        # 4) 정지
        self.get_logger().info('→ 정지: stop()')
        self.car.stop()

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
