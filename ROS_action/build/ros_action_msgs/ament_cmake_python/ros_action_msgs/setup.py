from setuptools import find_packages
from setuptools import setup

setup(
    name='ros_action_msgs',
    version='0.0.0',
    packages=find_packages(
        include=('ros_action_msgs', 'ros_action_msgs.*')),
)
