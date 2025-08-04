from setuptools import setup

package_name = 'mobile_vla_package'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        'h5py',
        'numpy',
        'opencv-python',
        'Pillow'
    ],
    zip_safe=True,
    maintainer='Mobile VLA Team',
    maintainer_email='soda@mobile-vla.io',
    description='Mobile VLA Data Collection System for RoboVLMs Action Chunks',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mobile_vla_collector = mobile_vla_package.mobile_vla_data_collector:main',
            'data_inspector = mobile_vla_package.data_inspector:main',
            'chunk_visualizer = mobile_vla_package.chunk_visualizer:main'
        ],
    },
)