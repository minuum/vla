import rclpy
from rclpy.node import Node
from serbot_interfaces.srv import Motor, Steering
import sys, tty, termios

DIRECTIONS = {
    'w': (1, 50),  # 전진
    's': (2, 50),  # 후진
    'x': (3, 0),   # 정지
}

STEERING = {
    'a': -1.0,  # 좌회전
    'd': 1.0,   # 우회전
    'z': 0.0,   # 직진
}

class KeyboardControl(Node):
    def __init__(self):
        super().__init__('keyboard_control')

        self.motor_cli = self.create_client(Motor, 'serbot/motor')
        self.steer_cli = self.create_client(Steering, 'serbot/steering')

        while not self.motor_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for motor service...')
        while not self.steer_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for steering service...')

        self.get_logger().info('Use keys: w/s/x for drive, a/d/z for steering')
        self.run()

    def get_key(self):
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        return ch

    def run(self):
        while True:
            key = self.get_key()

            if key in DIRECTIONS:
                direction, throttle = DIRECTIONS[key]
                req = Motor.Request()
                req.throttle = throttle
                req.direction = direction
                self.motor_cli.call_async(req)

            elif key in STEERING:
                req = Steering.Request()
                req.steering = STEERING[key]
                self.steer_cli.call_async(req)

def main(args=None):
    rclpy.init(args=args)
    node = KeyboardControl()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
