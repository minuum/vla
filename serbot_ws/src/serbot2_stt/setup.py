from setuptools import find_packages, setup

package_name = 'serbot2_stt'

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
            'mic_publisher_node = serbot2_stt.mic_publisher_node:main',
            'stt_subscriber_node = serbot2_stt.stt_subscriber_node:main',
        ],
    },
)
