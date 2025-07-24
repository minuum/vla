import rclpy, sys
from rclpy.node import Node
from serbot_interfaces.msg import BotAccelometer
from serbot_interfaces.msg import BotMagnetic
from serbot_interfaces.msg import BotGyroscope
from serbot_interfaces.msg import BotEuler
from serbot_interfaces.msg import BotQuaternion
from pop.Imu import Imu


class ImuPub(Node):
    def __init__(self, period=None):
        super().__init__("imu_publisher")
        self.imu = Imu()
        self.accel_publisher = self.create_publisher(
            BotAccelometer, "serbot/accelometer", 10
        )
        self.mag_publisher = self.create_publisher(BotMagnetic, "serbot/magnetic", 10)
        self.gyro_publisher = self.create_publisher(
            BotGyroscope, "serbot/gyroscope", 10
        )
        self.euler_publisher = self.create_publisher(BotEuler, "serbot/euler", 10)
        self.quat_publisher = self.create_publisher(
            BotQuaternion, "serbot/quaternion", 10
        )
        self.imu.callback(self.imu_cb)

    def imu_cb(self, data):
        msg = BotAccelometer()
        msg.x = float(data[0][0].tolist())
        msg.y = float(data[0][1].tolist())
        msg.z = float(data[0][2].tolist())
        self.accel_publisher.publish(msg)

        msg = BotMagnetic()
        msg.x = float(data[1][0].tolist())
        msg.y = float(data[1][1].tolist())
        msg.z = float(data[1][2].tolist())
        self.mag_publisher.publish(msg)

        msg = BotGyroscope()
        msg.x = float(data[2][0].tolist())
        msg.y = float(data[2][1].tolist())
        msg.z = float(data[2][2].tolist())
        self.gyro_publisher.publish(msg)

        msg = BotEuler()
        msg.yaw = float(data[3][0].tolist())
        msg.roll = float(data[3][1].tolist())
        msg.pitch = float(data[3][2].tolist())
        self.euler_publisher.publish(msg)

        msg = BotQuaternion()
        msg.w = float(data[4][0].tolist())
        msg.x = float(data[4][1].tolist())
        msg.y = float(data[4][2].tolist())
        msg.z = float(data[4][3].tolist())
        self.quat_publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    imu_publisher = ImuPub()
    try:
        rclpy.spin(imu_publisher)
    except KeyboardInterrupt:
        imu_publisher.imu.stop()
    imu_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
