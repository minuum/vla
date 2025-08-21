#!/usr/bin/env python3
"""
🚀 Mobile VLA System Launch File
카메라, 추론, 제어, 데이터 수집 노드들을 실행하는 통합 launch 파일
ROS sourcing 포함
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, EmitEvent
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from launch.event_handlers import OnProcessStart
from launch.events import matches_action


def generate_launch_description():
    """Mobile VLA 시스템 launch 파일 생성"""
    
    # ROS 환경 설정
    ros_setup = ExecuteProcess(
        cmd=['bash', '-c', 'source /opt/ros/humble/setup.bash && export RMW_IMPLEMENTATION=rmw_fastrtps_cpp && export ROS_DISTRO=humble && export LD_LIBRARY_PATH=/opt/ros/humble/lib:$LD_LIBRARY_PATH'],
        output='screen'
    )
    
    # Launch 인수들
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
    
    # 노드들
    nodes = []
    
    # 1. 카메라 노드 (camera_pub 패키지에서)
    if LaunchConfiguration('camera_node') == 'true':
        camera_node = Node(
            package='camera_pub',
            executable='camera_publisher_continuous.py',
            name='camera_publisher',
            output='screen',
            parameters=[{
                'camera_topic': '/camera/image_raw',
                'frame_rate': 10.0,
                'image_width': 640,
                'image_height': 480
            }],
            on_exit=EmitEvent(event=matches_action(camera_node))
        )
        nodes.append(camera_node)
    
    # 2. VLA 추론 노드
    if LaunchConfiguration('inference_node') == 'true':
        vla_inference_node = Node(
            package='mobile_vla_package',
            executable='vla_inference_node.py',
            name='vla_inference_node',
            output='screen',
            parameters=[{
                'model_path': '/workspace/vla/mobile_vla_dataset/models',
                'confidence_threshold': 0.7,
                'inference_rate': 1.0
            }]
        )
        nodes.append(vla_inference_node)
    
    # 3. 로봇 제어 노드
    if LaunchConfiguration('control_node') == 'true':
        robot_control_node = Node(
            package='mobile_vla_package',
            executable='robot_control_node.py',
            name='robot_control_node',
            output='screen',
            parameters=[{
                'control_mode': 'vla',  # 'manual', 'vla', 'hybrid'
                'max_linear_velocity': 0.5,
                'max_angular_velocity': 1.0,
                'cmd_vel_topic': '/cmd_vel'
            }]
        )
        nodes.append(robot_control_node)
    
    # 4. 데이터 수집 노드
    if LaunchConfiguration('data_collector') == 'true':
        data_collector_node = Node(
            package='mobile_vla_package',
            executable='mobile_vla_data_collector.py',
            name='mobile_vla_data_collector',
            output='screen',
            parameters=[{
                'data_save_path': '/workspace/vla/mobile_vla_dataset',
                'episode_duration': 30.0,
                'save_images': True,
                'save_actions': True
            }]
        )
        nodes.append(data_collector_node)
    
    # 5. ROS 환경 설정 노드
    ros_env_node = Node(
        package='mobile_vla_package',
        executable='ros_env_setup.py',
        name='ros_env_setup',
        output='screen',
        parameters=[{
            'rmw_implementation': 'rmw_fastrtps_cpp',
            'ros_distro': 'humble'
        }]
    )
    
    # Launch 설명 생성
    return LaunchDescription([
        ros_setup,
        camera_node_arg,
        inference_node_arg,
        control_node_arg,
        data_collector_arg,
        ros_env_node,
        *nodes
    ])
