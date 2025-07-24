import rclpy
from rclpy.node import Node
from serbot_interfaces.msg import BotUltrasonic
from serbot_interfaces.msg import BotPsd
from serbot_interfaces.srv import Motor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from threading import Thread


class GoFront(Node):
    def __init__(self):
        super().__init__("go_front")
        self.callback_group = ReentrantCallbackGroup()
        self.subscription_ult = self.create_subscription(
            BotUltrasonic,
            "serbot/ultrasonic",
            self.ultrasonic_callback,
            1,
            callback_group=self.callback_group,
        )

        self.subscription_psd = self.create_subscription(
            BotPsd,
            "serbot/psd",
            self.psd_callback,
            1,
            callback_group=self.callback_group,
        )

        self.cli = self.create_client(
            Motor, "serbot/motor", callback_group=self.callback_group
        )
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("service not available, waiting again...")
        self.motor = Motor.Request()
        self.motor.direction = 3
        self.distance = [10, 10, 10]
        self.timer = self.create_timer(
            0.1, self.send_request, callback_group=self.callback_group
        )

    def send_request(self):
        if 30 < min(self.distance):
            self.motor.direction = 1
            self.motor.throttle = 20
        else:
            self.motor.direction = 3
            self.motor.throttle = 0

        self.get_logger().info("Request direction: %d" % self.motor.direction)
        self.future = self.cli.call_async(self.motor)
        try:
            response = self.future.result()

            if response is not None:
                if response.result == 0:
                    self.get_logger().info("Working succeed")
            else:
                self.get_logger().info("Working failed")
        except Exception as e:
            self.get_logger().info("Service call failed %r" % (e,))

    def ultrasonic_callback(self, msg):
        self.get_logger().info("%r" % msg)

        self.distance[1:3] = msg.ultra[:2]

    def psd_callback(self, msg):
        self.get_logger().info("%r" % msg)

        self.distance[0] = msg.psd[0]


def main(args=None):
    rclpy.init(args=args)
    try:
        go_front = GoFront()
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(go_front)
        try:
            executor.spin()
        except KeyboardInterrupt:
            go_front.get_logger().info("Keyboard Interrupt (SIGINT)")
        finally:
            executor.shutdown()
            go_front.destroy_node()
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
