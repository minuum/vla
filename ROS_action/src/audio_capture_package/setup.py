from setuptools import find_packages, setup

package_name = 'audio_capture_package'

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
    maintainer_email='soda@example.com',
    description='A package to capture audio from microphone and publish it.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'audio_capture_node = audio_capture_package.mic_publisher_node:main',
        ],
    },
)
