import rclpy, sys
from rclpy.node import Node
from serbot_interfaces.msg import BotBattery
from pop.Battery import Battery


class BatteryPub(Node):
    def __init__(self, period=None):
        super().__init__("battery_publisher")
        self.batt = Battery()
        self.publisher = self.create_publisher(BotBattery, "serbot/battery", 10)
        self.batt.callback(self.batt_cb)

    def batt_cb(self, data):
        msg = BotBattery()
        msg.volt = data[0]
        msg.ntc = data[1]
        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    batt_publisher = BatteryPub()
    try:
        rclpy.spin(batt_publisher)
    except KeyboardInterrupt:
        batt_publisher.batt.stop()
    batt_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
