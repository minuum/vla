#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os

def generate_launch_description():
    # Launch 인자들
    camera_node_arg = DeclareLaunchArgument(
        'camera_node',
        default_value='true',
        description='카메라 노드 실행 여부'
    )
    
    inference_node_arg = DeclareLaunchArgument(
        'inference_node',
        default_value='true',
        description='VLA 추론 노드 실행 여부'
    )
    
    control_node_arg = DeclareLaunchArgument(
        'control_node',
        default_value='true',
        description='로봇 제어 노드 실행 여부'
    )
    
    data_collector_arg = DeclareLaunchArgument(
        'data_collector',
        default_value='false',
        description='데이터 수집 노드 실행 여부'
    )
    
    # 패키지 경로 (직접 지정)
    camera_pub_pkg = '/workspace/vla/ROS_action/install/camera_pub'
    mobile_vla_pkg = '/workspace/vla/ROS_action/install/mobile_vla_package'
    
    # 노드들
    nodes = []
    
    # 1. 카메라 노드 (항상 먼저 시작)
    if LaunchConfiguration('camera_node') == 'true':
        camera_node = Node(
            package='camera_pub',
            executable='camera_publisher_continuous',
            name='camera_service_server',
            output='screen',
            parameters=[{
                'camera_resolution': '1280x720',
                'camera_fps': 30,
                'buffer_size': 1
            }]
        )
        nodes.append(camera_node)
    
    # 2. VLA 추론 노드 (카메라 후 시작)
    if LaunchConfiguration('inference_node') == 'true':
        inference_node = Node(
            package='mobile_vla_package',
            executable='vla_inference_node',
            name='vla_inference_node',
            output='screen',
            parameters=[{
                'inference_interval': 0.5,
                'confidence_threshold': 0.7,
                'model_name': 'microsoft/kosmos-2-patch14-224'
            }]
        )
        # 카메라 노드 후 3초 뒤 시작
        delayed_inference = TimerAction(
            period=3.0,
            actions=[inference_node]
        )
        nodes.append(delayed_inference)
    
    # 3. 로봇 제어 노드
    if LaunchConfiguration('control_node') == 'true':
        control_node = Node(
            package='mobile_vla_package',
            executable='robot_control_node',
            name='robot_control_node',
            output='screen',
            parameters=[{
                'control_mode': 'manual',
                'manual_priority': True,
                'throttle': 50
            }]
        )
        nodes.append(control_node)
    
    # 4. 데이터 수집 노드 (선택적)
    if LaunchConfiguration('data_collector') == 'true':
        data_collector_node = Node(
            package='mobile_vla_package',
            executable='mobile_vla_data_collector',
            name='mobile_vla_data_collector',
            output='screen',
            parameters=[{
                'episode_length': 18,
                'action_chunk_size': 8
            }]
        )
        nodes.append(data_collector_node)
    
    return LaunchDescription([
        camera_node_arg,
        inference_node_arg,
        control_node_arg,
        data_collector_arg,
        *nodes
    ])
