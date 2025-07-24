from launch_ros.actions import Node
from launch import LaunchDescription


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="serbot_ros",
                executable="driving_service",
                name="driving_service",
            ),
            Node(package="serbot_ros", executable="ultrasonic", name="ultrasonic"),
            Node(package="serbot_ros", executable="psd", name="psd"),
            Node(package="serbot_ros", executable="imu", name="imu"),
            Node(package="serbot_ros", executable="light", name="light"),
            Node(package="serbot_ros", executable="battery", name="battery"),
        ]
    )
