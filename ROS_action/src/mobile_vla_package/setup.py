import os
import glob
from setuptools import find_packages, setup

package_name = 'mobile_vla_package'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob.glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='soda',
    maintainer_email='soda@example.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    #tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vla_collector = mobile_vla_package.mobile_vla_data_collector:main',
            'simple_robot_mover = mobile_vla_package.simple_robot_mover:main',
            'mobile_vla_inference = mobile_vla_package.mobile_vla_inference:main',
            'action_sequence_executor = mobile_vla_package.action_sequence_executor:main',
            'system_monitor = mobile_vla_package.system_monitor:main',
            'test_camera_simulator = mobile_vla_package.test_camera_simulator:main',
            'test_monitor = mobile_vla_package.test_monitor:main',
            'simple_inference_test = mobile_vla_package.simple_inference_test:main',
            'robovlms_inference = mobile_vla_package.robovlms_inference:main',
            'robovlms_controller = mobile_vla_package.robovlms_controller:main',
            'robovlms_monitor = mobile_vla_package.robovlms_monitor:main',
        ]
    },
)
