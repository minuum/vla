#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # 패키지 디렉토리 가져오기
    pkg_share = get_package_share_directory('mobile_vla_package')
    
    # Launch 인수 선언
    camera_node_arg = DeclareLaunchArgument(
        'camera_node',
        default_value='true',
        description='카메라 노드 실행 여부'
    )
    
    inference_node_arg = DeclareLaunchArgument(
        'inference_node',
        default_value='false',
        description='추론 노드 실행 여부'
    )
    
    control_node_arg = DeclareLaunchArgument(
        'control_node',
        default_value='false',
        description='제어 노드 실행 여부'
    )
    
    data_collector_arg = DeclareLaunchArgument(
        'data_collector',
        default_value='false',
        description='데이터 수집기 실행 여부'
    )
    
    # 노드 정의
    camera_node = Node(
        package='camera_pub',
        executable='camera_publisher',
        name='camera_publisher',
        output='screen',
        condition=LaunchConfiguration('camera_node')
    )
    
    inference_node = Node(
        package='mobile_vla_package',
        executable='mobile_vla_inference',
        name='mobile_vla_inference',
        output='screen',
        condition=LaunchConfiguration('inference_node')
    )
    
    action_executor_node = Node(
        package='mobile_vla_package',
        executable='action_sequence_executor',
        name='action_sequence_executor',
        output='screen',
        condition=LaunchConfiguration('inference_node')
    )
    
    system_monitor_node = Node(
        package='mobile_vla_package',
        executable='system_monitor',
        name='system_monitor',
        output='screen',
        condition=LaunchConfiguration('inference_node')
    )
    
    control_node = Node(
        package='omni_controller',
        executable='omni_controller',
        name='omni_controller',
        output='screen',
        condition=LaunchConfiguration('control_node')
    )
    
    data_collector_node = Node(
        package='mobile_vla_package',
        executable='mobile_vla_data_collector',
        name='mobile_vla_data_collector',
        output='screen',
        condition=LaunchConfiguration('data_collector')
    )
    
    return LaunchDescription([
        camera_node_arg,
        inference_node_arg,
        control_node_arg,
        data_collector_arg,
        camera_node,
        inference_node,
        action_executor_node,
        system_monitor_node,
        control_node,
        data_collector_node
    ])
