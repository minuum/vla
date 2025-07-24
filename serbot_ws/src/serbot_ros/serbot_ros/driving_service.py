import rclpy, math
from rclpy.node import Node
from serbot_interfaces.srv import Motor
from serbot_interfaces.srv import Steering
from pop.driving import Driving
from sensor_msgs.msg import JointState


class DrivingService(Node):
    def __init__(self):
        super().__init__("driving_service")
        self.drv = Driving()
        self.motor_srv = self.create_service(Motor, "serbot/motor", self.motor_cb)
        self.steering_srv = self.create_service(
            Steering, "serbot/steering", self.steering_cb
        )
        self.get_logger().info("driving service is activated")
        self.throttle = 0
        self.steering = 0
        self.direction = 0
        self.joint_steering = 0
        self.joint_wheel = 0
        self.joint_state = JointState()
        self.joint_state.name = ["wheel_joint1", "wheel_joint2", "wheel_joint3"]
        self.joint_publisher = self.create_publisher(JointState, "joint_states", 10)
        self.joint_timer = self.create_timer(0.03, self.joint_cb)

    def motor_cb(self, request, response):
        response.result = 0
        self.throttle = request.throttle
        self.direction = request.direction
        if self.direction == 1:
            self.drv.forward(self.throttle)
        elif self.direction == 2:
            self.drv.backward(self.throttle)
        elif self.direction == 3:
            self.throttle = 0
            self.drv.stop()
        else:
            self.get_logger().info("direction request failed..invalid data")
            response.result = -1

        return response

    def steering_cb(self, request, response):
        if request.steering > 1 or request.steering < -1:
            self.get_logger().info("steering request failed..invalid data")
            response.result = -1
        else:
            self.drv.steering = request.steering
            self.steering = request.steering
            response.result = 0

        return response

    def joint_cb(self):
        self.joint_state.header.stamp = self.get_clock().now().to_msg()
        if self.direction == 1:
            self.joint_wheel = self.joint_wheel + self.throttle / 100
        elif self.direction == 2:
            self.joint_wheel = self.joint_wheel - self.throttle / 100

        self.joint_steering = self.joint_steering + self.steering * 1.5

        self.joint_state.position = [
            float(self.joint_wheel),
            float(self.joint_wheel),
            float(self.joint_steering),
        ]
        self.joint_publisher.publish(self.joint_state)


def main(args=None):
    rclpy.init(args=args)
    driving_service = DrivingService()
    rclpy.spin(driving_service)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
