import rclpy, sys
from rclpy.node import Node
from serbot_interfaces.msg import BotLight
from pop.Light import Light


class LightPub(Node):
    def __init__(self, period=None):
        super().__init__("light_publisher")
        self.light = Light()
        self.publisher = self.create_publisher(BotLight, "serbot/light", 10)
        self.light.callback(self.light_cb)

    def light_cb(self, data):
        msg = BotLight()
        msg.light = data
        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    light_publisher = LightPub()
    try:
        rclpy.spin(light_publisher)
    except KeyboardInterrupt:
        light_publisher.light.stop()
    light_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
