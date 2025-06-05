#omni_drive_node.py
import rclpy
from rclpy.node import Node
from pop.driving import Driving
from pop.Psd import Psd
from pop.Ultrasonic import Ultrasonic
from ros_action_msgs.msg import ObjectInfo
import time

class OmniDriveNode(Node):
    def __init__(self):
        super().__init__('omni_drive_node')
        self.car = Driving()

        # 센서 초기화
        self.psd = Psd(dev="can0", bitrate=500000)
        self.us = Ultrasonic(dev="can0", bitrate=500000)

        # 목표 각도 저장
        self.target_angle = None
        
        # ObjectInfo 구독 (angle만 사용)
        self.create_subscription(
            ObjectInfo,
            '/object_info',
            self.object_info_callback,
            1
        )

        self.create_timer(0.1, self.control_loop)

    def object_info_callback(self, msg: ObjectInfo):
        self.target_angle = msg.angle
        self.get_logger().info(f'수신 각도: {self.target_angle:.1f}°')

 
    def control_loop(self):
        # 이동 중 장애물 감지 체크
        if self.target_angle is None:
            if getattr(self, 'moving', False):
                psd_min = min(self.psd.read())
                us_min = min(self.us.read())
                if psd_min <= 20 or us_min <= 20:
                    self.get_logger().info(f'(PSD={psd_min}, US={us_min}) → 정지')
                    self.car.stop()
                    self.moving = False
            return

        # 각도 변환: 0~180도는 그대로, 180~360도는 (angle-360)
        angle = self.target_angle
        if angle > 180:
            angle = angle - 360  # 예: 340도 → -20도

        # spin_speed: 양수(좌측, 반시계방향) → -30, 음수(우측, 시계방향) → 30
        spin_speed = -30 if angle >= 0 else 30
        spin_time = abs(angle) / 92.0

        self.get_logger().info(
            f'회전 명령 → 속도={spin_speed}, 시간={spin_time:.2f}s (요청 각도={self.target_angle:.1f}°, 변환 각도={angle:.1f}°)'
        )
        self.car.spin(spin_speed)
        time.sleep(spin_time)
        self.car.stop()

        # 회전 후 전방으로 이동
        self.get_logger().info('이동 명령 → 각도=0°, 속도=20')
        self.car.move(0, 20)
        self.moving = True  # 이동 상태 플래그

        # 각도 초기화 → 반복 방지
        self.target_angle = None


def main(args=None):
    rclpy.init(args=args)
    node = OmniDriveNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
