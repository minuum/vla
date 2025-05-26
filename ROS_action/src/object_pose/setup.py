from setuptools import find_packages, setup
from glob import glob
import os
package_name = 'object_pose'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'resource'), glob('resource/*.png')),  
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='shosae',
    maintainer_email='shosae@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'object_pose_publisher = object_pose.object_pose_publisher:main',
            'cup_publisher = object_pose.cup_publisher:main'
        ],
    },
)
