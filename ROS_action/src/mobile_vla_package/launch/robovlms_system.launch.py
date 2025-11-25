#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.conditions import IfCondition

def generate_launch_description():
    """🏆 RoboVLMs 시스템 launch 파일 (SOTA 모델 사용)"""
    
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
    
    # 🏆 추론 모드 파라미터 (SOTA 모델 기본값)
    inference_mode_arg = DeclareLaunchArgument(
        'inference_mode',
        default_value='transformers',
        description='Inference mode: transformers (SOTA) or quantized'
    )
    
    model_type_arg = DeclareLaunchArgument(
        'model_type',
        default_value='kosmos2_clip_hybrid',  # 🏆 최고 성능 모델
        description='Quantized model type: kosmos2_clip_hybrid (SOTA), kosmos2_pure, kosmos2_simple, cpu_mae0222'
    )
    
    device_arg = DeclareLaunchArgument(
        'device',
        default_value='auto',
        description='Device type: auto, gpu, cpu'
    )
    
    # 🚀 Jetson 최적화 파라미터 추가
    optimization_mode_arg = DeclareLaunchArgument(
        'optimization_mode',
        default_value='auto',
        description='Jetson optimization mode: auto, tensorrt, fp16, int8, test'
    )
    
    memory_limit_arg = DeclareLaunchArgument(
        'memory_limit_gb',
        default_value='12.0',
        description='Memory limit in GB for auto mode'
    )
    
    enable_tensorrt_arg = DeclareLaunchArgument(
        'enable_tensorrt',
        default_value='true',
        description='Enable TensorRT optimization'
    )
    
    enable_quantization_arg = DeclareLaunchArgument(
        'enable_quantization',
        default_value='true',
        description='Enable quantization optimization'
    )
    
    # 노드 정의
    camera_simulator_node = Node(
        package='mobile_vla_package',
        executable='test_camera_simulator',
        name='camera_simulator',
        output='screen',
        condition=IfCondition(LaunchConfiguration('camera_simulator'))
    )
    
    # 🏆 RoboVLMs 추론 노드 (SOTA 모델)
    robovlms_inference_node = Node(
        package='mobile_vla_package',
        executable='robovlms_inference',
        name='robovlms_inference',
        output='screen',
        parameters=[{
            'inference_mode': LaunchConfiguration('inference_mode'),
            'model_type': LaunchConfiguration('model_type'),
            'device': LaunchConfiguration('device'),
            'optimization_mode': LaunchConfiguration('optimization_mode'),
            'memory_limit_gb': LaunchConfiguration('memory_limit_gb'),
            'enable_tensorrt': LaunchConfiguration('enable_tensorrt'),
            'enable_quantization': LaunchConfiguration('enable_quantization')
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
            'ros2', 'topic', 'pub', '/mobile_vla/system_control', 'std_msgs/msg/String',
            '"data: start"'
        ],
        output='screen'
    )
    
    return LaunchDescription([
        # Launch 인수
        camera_simulator_arg,
        robovlms_inference_arg,
        robovlms_controller_arg,
        robovlms_monitor_arg,
        inference_mode_arg,
        model_type_arg,
        device_arg,
        optimization_mode_arg,
        memory_limit_arg,
        enable_tensorrt_arg,
        enable_quantization_arg,
        
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
