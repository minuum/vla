from setuptools import setup
import os
from glob import glob

package_name = "serbot_ros"

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            os.path.join("share", package_name, "launch"),
            glob(os.path.join("launch", "*launch.[pxy][yma]*")),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="soda",
    maintainer_email="soda@soda.cc",
    description="SerBot ROS Package",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "battery = serbot_ros.battery:main",
            "light = serbot_ros.light:main",
            "ultrasonic = serbot_ros.ultrasonic:main",
            "psd = serbot_ros.psd:main",
            "imu = serbot_ros.imu:main",
            "driving_param = serbot_ros.driving_param:main",
            "driving_service = serbot_ros.driving_service:main",
        ],
    },
)
