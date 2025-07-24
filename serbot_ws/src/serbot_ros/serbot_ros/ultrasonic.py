import rclpy, sys
from rclpy.node import Node
from serbot_interfaces.msg import BotUltrasonic
from pop.Ultrasonic import Ultrasonic


class UltrasonicPub(Node):
    def __init__(self):
        super().__init__("ultrasonic_publisher")
        self.ultra = Ultrasonic()
        self.publisher = self.create_publisher(BotUltrasonic, "serbot/ultrasonic", 10)
        self.ultra.callback(self.ultra_cb)

    def ultra_cb(self, data):
        msg = BotUltrasonic()
        msg.ultra = [x.tolist() for x in data]
        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    ultra_publisher = UltrasonicPub()
    try:
        rclpy.spin(ultra_publisher)
    except KeyboardInterrupt:
        ultra_publisher.ultra.stop()
    ultra_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
