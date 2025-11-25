#!/usr/bin/env python3
"""
🔧 ROS Environment Setup Node
ROS2 환경 변수와 라이브러리 경로를 설정하는 노드
"""

import os
import sys
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter


class ROSEnvSetupNode(Node):
    """ROS 환경 설정 노드"""
    
    def __init__(self):
        super().__init__('ros_env_setup')
        
        # ROS 환경 설정
        self.setup_ros_environment()
        
        # 파라미터 설정
        self.declare_parameter('rmw_implementation', 'rmw_fastrtps_cpp')
        self.declare_parameter('ros_distro', 'humble')
        
        self.get_logger().info('🔧 ROS 환경 설정 노드 시작')
        
        # 환경 변수 설정
        rmw_impl = self.get_parameter('rmw_implementation').value
        ros_distro = self.get_parameter('ros_distro').value
        
        os.environ['RMW_IMPLEMENTATION'] = rmw_impl
        os.environ['ROS_DISTRO'] = ros_distro
        os.environ['LD_LIBRARY_PATH'] = f'/opt/ros/{ros_distro}/lib:{os.environ.get("LD_LIBRARY_PATH", "")}'
        
        self.get_logger().info(f'✅ RMW_IMPLEMENTATION: {rmw_impl}')
        self.get_logger().info(f'✅ ROS_DISTRO: {ros_distro}')
        self.get_logger().info(f'✅ LD_LIBRARY_PATH: {os.environ["LD_LIBRARY_PATH"]}')
        
        # ROS2 명령어 테스트
        self.test_ros2_commands()
    
    def setup_ros_environment(self):
        """ROS 환경 설정"""
        try:
            # ROS2 setup.bash 소스
            ros_setup_path = '/opt/ros/humble/setup.bash'
            if os.path.exists(ros_setup_path):
                # 환경 변수 설정
                os.environ['ROS_DISTRO'] = 'humble'
                os.environ['RMW_IMPLEMENTATION'] = 'rmw_fastrtps_cpp'
                os.environ['LD_LIBRARY_PATH'] = '/opt/ros/humble/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
                
                self.get_logger().info('✅ ROS2 환경 설정 완료')
            else:
                self.get_logger().error(f'❌ ROS2 setup.bash를 찾을 수 없습니다: {ros_setup_path}')
                
        except Exception as e:
            self.get_logger().error(f'❌ ROS 환경 설정 중 오류: {str(e)}')
    
    def test_ros2_commands(self):
        """ROS2 명령어 테스트"""
        try:
            import subprocess
            
            # ros2 --help 테스트
            result = subprocess.run(['ros2', '--help'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                self.get_logger().info('✅ ROS2 명령어 정상 작동')
            else:
                self.get_logger().warning(f'⚠️ ROS2 명령어 오류: {result.stderr}')
                
        except Exception as e:
            self.get_logger().warning(f'⚠️ ROS2 명령어 테스트 실패: {str(e)}')


def main(args=None):
    """메인 함수"""
    rclpy.init(args=args)
    
    try:
        node = ROSEnvSetupNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'❌ 노드 실행 중 오류: {str(e)}')
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
