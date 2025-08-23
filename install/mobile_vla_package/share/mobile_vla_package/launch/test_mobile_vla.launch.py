#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # 패키지 디렉토리 가져오기
    pkg_share = get_package_share_directory('mobile_vla_package')
    
    # Launch 인수 선언
    use_camera_simulator_arg = DeclareLaunchArgument(
        'use_camera_simulator',
        default_value='true',
        description='카메라 시뮬레이터 사용 여부'
    )
    
    use_test_monitor_arg = DeclareLaunchArgument(
        'use_test_monitor',
        default_value='true',
        description='테스트 모니터 사용 여부'
    )
    
    # 테스트 카메라 시뮬레이터 노드
    camera_simulator_node = Node(
        package='mobile_vla_package',
        executable='test_camera_simulator',
        name='test_camera_simulator',
        output='screen',
        condition=LaunchConfiguration('use_camera_simulator')
    )
    
    # Mobile VLA 추론 노드
    inference_node = Node(
        package='mobile_vla_package',
        executable='mobile_vla_inference',
        name='mobile_vla_inference',
        output='screen'
    )
    
    # 액션 시퀀스 실행 노드
    action_executor_node = Node(
        package='mobile_vla_package',
        executable='action_sequence_executor',
        name='action_sequence_executor',
        output='screen'
    )
    
    # 시스템 모니터링 노드
    system_monitor_node = Node(
        package='mobile_vla_package',
        executable='system_monitor',
        name='system_monitor',
        output='screen'
    )
    
    # 테스트 모니터 노드
    test_monitor_node = Node(
        package='mobile_vla_package',
        executable='test_monitor',
        name='test_monitor',
        output='screen',
        condition=LaunchConfiguration('use_test_monitor')
    )
    
    return LaunchDescription([
        use_camera_simulator_arg,
        use_test_monitor_arg,
        camera_simulator_node,
        inference_node,
        action_executor_node,
        system_monitor_node,
        test_monitor_node
    ])
