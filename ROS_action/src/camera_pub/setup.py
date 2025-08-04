from setuptools import find_packages, setup

package_name = 'camera_pub'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
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
            'camera_publisher_continuous = camera_pub.camera_publisher_continuous:main',
            'camera_publisher_usb = camera_pub.camera_publisher_usb:main'
        ],
    },
)
