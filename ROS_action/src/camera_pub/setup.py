from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'camera_pub'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(where='.', exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'srv'), glob('srv/*.srv')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='soda',
    maintainer_email='soda@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_publisher_node = camera_pub.camera_publisher_node:main',
            'camera_publisher_usb = camera_pub.camera_publisher_usb:main',
            'camera_service_server = camera_pub.camera_publisher_continuous:main',
            'usb_camera_service_server = camera_pub.camera_publisher_usb_service:main',
        ],
    },
)

