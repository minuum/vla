import rclpy, sys
from rclpy.node import Node
from serbot_interfaces.msg import BotPsd
from pop.Psd import Psd


class PsdPub(Node):
    def __init__(self):
        super().__init__("psd_publisher")
        self.psd = Psd()
        self.publisher = self.create_publisher(BotPsd, "serbot/psd", 10)
        self.psd.callback(self.psd_cb)

    def psd_cb(self, data):
        msg = BotPsd()
        msg.psd = [x.tolist() for x in data]
        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    psd_publisher = PsdPub()
    try:
        rclpy.spin(psd_publisher)
    except KeyboardInterrupt:
        psd_publisher.psd.stop()
    psd_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
