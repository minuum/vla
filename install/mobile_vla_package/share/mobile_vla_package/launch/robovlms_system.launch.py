#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.conditions import IfCondition

def generate_launch_description():
    """RoboVLMs 시스템 launch 파일"""
    
    # Launch 인수 선언
    camera_simulator_arg = DeclareLaunchArgument(
        'camera_simulator',
        default_value='true',
        description='Enable camera simulator'
    )
    
    robovlms_inference_arg = DeclareLaunchArgument(
        'robovlms_inference',
        default_value='true',
        description='Enable RoboVLMs inference node'
    )
    
    robovlms_controller_arg = DeclareLaunchArgument(
        'robovlms_controller',
        default_value='true',
        description='Enable RoboVLMs controller'
    )
    
    robovlms_monitor_arg = DeclareLaunchArgument(
        'robovlms_monitor',
        default_value='true',
        description='Enable RoboVLMs monitor'
    )
    
    # 노드 정의
    camera_simulator_node = Node(
        package='mobile_vla_package',
        executable='test_camera_simulator',
        name='camera_simulator',
        output='screen',
        condition=IfCondition(LaunchConfiguration('camera_simulator'))
    )
    
    robovlms_inference_node = Node(
        package='mobile_vla_package',
        executable='robovlms_inference',
        name='robovlms_inference',
        output='screen',
        parameters=[{
            'model_name': 'minium/mobile-vla-omniwheel',  # MAE 0.222 달성한 최신 모델
            'device': 'auto'
        }],
        condition=IfCondition(LaunchConfiguration('robovlms_inference'))
    )
    
    robovlms_controller_node = Node(
        package='mobile_vla_package',
        executable='robovlms_controller',
        name='robovlms_controller',
        output='screen',
        condition=IfCondition(LaunchConfiguration('robovlms_controller'))
    )
    
    robovlms_monitor_node = Node(
        package='mobile_vla_package',
        executable='robovlms_monitor',
        name='robovlms_monitor',
        output='screen',
        condition=IfCondition(LaunchConfiguration('robovlms_monitor'))
    )
    
    # 시스템 시작 명령
    start_system_cmd = ExecuteProcess(
        cmd=[
            'ros2', 'topic', 'pub', '/robovlms/user_command', 'std_msgs/msg/String',
            '{"action": "start"}'
        ],
        output='screen'
    )
    
    return LaunchDescription([
        # Launch 인수
        camera_simulator_arg,
        robovlms_inference_arg,
        robovlms_controller_arg,
        robovlms_monitor_arg,
        
        # 노드들
        camera_simulator_node,
        robovlms_inference_node,
        robovlms_controller_node,
        robovlms_monitor_node,
        
        # 시스템 시작 (5초 후)
        ExecuteProcess(
            cmd=['sleep', '5'],
            output='screen'
        ),
        start_system_cmd
    ])
