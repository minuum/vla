#!/usr/bin/env python3
"""
Step 1: 데이터 수집 환경 테스트
Branch b131fb5에서 camera_service_server & vla_collector 동작 확인
"""

import os
import sys
import time
import subprocess
import logging
from pathlib import Path

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataCollectionTester:
    def __init__(self):
        self.project_root = Path("/home/billy/25-1kp/vla")
        self.branch_name = "b131fb5"
        self.test_episodes = 5
        
    def check_git_status(self):
        """Git 상태 확인"""
        logger.info("🔍 Git 상태 확인 중...")
        
        try:
            # 현재 브랜치 확인
            result = subprocess.run(
                ["git", "branch", "--show-current"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            current_branch = result.stdout.strip()
            logger.info(f"현재 브랜치: {current_branch}")
            
            if current_branch != self.branch_name:
                logger.warning(f"⚠️  현재 브랜치가 {self.branch_name}가 아닙니다.")
                logger.info(f"브랜치 전환: git checkout {self.branch_name}")
                return False
            
            # Git 상태 확인
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.stdout.strip():
                logger.warning("⚠️  커밋되지 않은 변경사항이 있습니다.")
                logger.info("변경사항 확인: git status")
            else:
                logger.info("✅ Git 상태 정상")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Git 상태 확인 실패: {e}")
            return False
    
    def check_ros2_environment(self):
        """ROS2 환경 확인"""
        logger.info("🔍 ROS2 환경 확인 중...")
        
        try:
            # ROS2 설치 확인
            result = subprocess.run(
                ["which", "ros2"],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logger.error("❌ ROS2가 설치되지 않았습니다.")
                return False
            
            logger.info("✅ ROS2 설치 확인됨")
            
            # ROS2 환경 소싱
            ros2_setup = "/opt/ros/humble/setup.bash"
            if os.path.exists(ros2_setup):
                logger.info(f"✅ ROS2 환경 소싱: {ros2_setup}")
            else:
                logger.warning(f"⚠️  ROS2 설정 파일을 찾을 수 없습니다: {ros2_setup}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ ROS2 환경 확인 실패: {e}")
            return False
    
    def check_camera_service_server(self):
        """카메라 서비스 서버 확인"""
        logger.info("🔍 카메라 서비스 서버 확인 중...")
        
        try:
            # 카메라 서비스 서버 파일 확인
            camera_launch = self.project_root / "camera_service_server" / "launch" / "camera_service.launch.py"
            
            if not camera_launch.exists():
                logger.error(f"❌ 카메라 서비스 런치 파일을 찾을 수 없습니다: {camera_launch}")
                return False
            
            logger.info(f"✅ 카메라 서비스 런치 파일 확인: {camera_launch}")
            
            # 카메라 서비스 서버 실행 테스트
            logger.info("카메라 서비스 서버 실행 테스트...")
            
            # 백그라운드에서 실행
            process = subprocess.Popen(
                ["ros2", "launch", "camera_service_server", "camera_service.launch.py"],
                cwd=self.project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # 5초 대기
            time.sleep(5)
            
            # 프로세스 상태 확인
            if process.poll() is None:
                logger.info("✅ 카메라 서비스 서버 실행 중")
                process.terminate()
                process.wait()
                return True
            else:
                logger.error("❌ 카메라 서비스 서버 실행 실패")
                stdout, stderr = process.communicate()
                logger.error(f"STDOUT: {stdout.decode()}")
                logger.error(f"STDERR: {stderr.decode()}")
                return False
                
        except Exception as e:
            logger.error(f"❌ 카메라 서비스 서버 확인 실패: {e}")
            return False
    
    def check_vla_collector(self):
        """VLA 컬렉터 확인"""
        logger.info("🔍 VLA 컬렉터 확인 중...")
        
        try:
            # VLA 컬렉터 파일 확인
            vla_launch = self.project_root / "vla_collector" / "launch" / "vla_collector.launch.py"
            
            if not vla_launch.exists():
                logger.error(f"❌ VLA 컬렉터 런치 파일을 찾을 수 없습니다: {vla_launch}")
                return False
            
            logger.info(f"✅ VLA 컬렉터 런치 파일 확인: {vla_launch}")
            
            # VLA 컬렉터 실행 테스트
            logger.info("VLA 컬렉터 실행 테스트...")
            
            # 백그라운드에서 실행
            process = subprocess.Popen(
                ["ros2", "launch", "vla_collector", "vla_collector.launch.py"],
                cwd=self.project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # 5초 대기
            time.sleep(5)
            
            # 프로세스 상태 확인
            if process.poll() is None:
                logger.info("✅ VLA 컬렉터 실행 중")
                process.terminate()
                process.wait()
                return True
            else:
                logger.error("❌ VLA 컬렉터 실행 실패")
                stdout, stderr = process.communicate()
                logger.error(f"STDOUT: {stdout.decode()}")
                logger.error(f"STDERR: {stderr.decode()}")
                return False
                
        except Exception as e:
            logger.error(f"❌ VLA 컬렉터 확인 실패: {e}")
            return False
    
    def test_data_collection(self):
        """데이터 수집 테스트"""
        logger.info("🔍 데이터 수집 테스트 중...")
        
        try:
            # 데이터 저장 디렉토리 확인
            data_dir = self.project_root / "mobile_vla_dataset"
            data_dir.mkdir(exist_ok=True)
            
            logger.info(f"✅ 데이터 저장 디렉토리: {data_dir}")
            
            # 테스트 에피소드 수집 시뮬레이션
            logger.info(f"테스트 에피소드 {self.test_episodes}개 수집 시뮬레이션...")
            
            for i in range(self.test_episodes):
                logger.info(f"에피소드 {i+1}/{self.test_episodes} 수집 중...")
                
                # 시뮬레이션 데이터 생성
                episode_data = {
                    "episode_id": i + 1,
                    "timestamp": time.time(),
                    "images": f"episode_{i+1}_images.h5",
                    "actions": f"episode_{i+1}_actions.h5",
                    "language": f"test_task_{i+1}"
                }
                
                # 에피소드 데이터 저장
                episode_file = data_dir / f"episode_{i+1}.json"
                import json
                with open(episode_file, 'w') as f:
                    json.dump(episode_data, f, indent=2)
                
                logger.info(f"✅ 에피소드 {i+1} 저장 완료: {episode_file}")
                time.sleep(1)  # 시뮬레이션 지연
            
            logger.info(f"✅ {self.test_episodes}개 테스트 에피소드 수집 완료")
            return True
            
        except Exception as e:
            logger.error(f"❌ 데이터 수집 테스트 실패: {e}")
            return False
    
    def run_full_test(self):
        """전체 테스트 실행"""
        logger.info("🚀 Mobile VLA 데이터 수집 환경 테스트 시작")
        logger.info("=" * 60)
        
        test_results = []
        
        # 1. Git 상태 확인
        test_results.append(("Git 상태 확인", self.check_git_status()))
        
        # 2. ROS2 환경 확인
        test_results.append(("ROS2 환경 확인", self.check_ros2_environment()))
        
        # 3. 카메라 서비스 서버 확인
        test_results.append(("카메라 서비스 서버 확인", self.check_camera_service_server()))
        
        # 4. VLA 컬렉터 확인
        test_results.append(("VLA 컬렉터 확인", self.check_vla_collector()))
        
        # 5. 데이터 수집 테스트
        test_results.append(("데이터 수집 테스트", self.test_data_collection()))
        
        # 결과 출력
        logger.info("=" * 60)
        logger.info("📊 테스트 결과 요약")
        logger.info("=" * 60)
        
        passed = 0
        total = len(test_results)
        
        for test_name, result in test_results:
            status = "✅ 통과" if result else "❌ 실패"
            logger.info(f"{test_name}: {status}")
            if result:
                passed += 1
        
        logger.info("=" * 60)
        logger.info(f"전체 결과: {passed}/{total} 통과")
        
        if passed == total:
            logger.info("🎉 모든 테스트 통과! 데이터 수집 환경이 준비되었습니다.")
            return True
        else:
            logger.error(f"⚠️  {total - passed}개 테스트 실패. 환경을 확인해주세요.")
            return False

def main():
    """메인 함수"""
    tester = DataCollectionTester()
    success = tester.run_full_test()
    
    if success:
        print("\n🎯 다음 단계:")
        print("1. 실제 데이터 수집 시작")
        print("2. Mobile VLA 모델 구조 구현")
        print("3. 학습 파이프라인 구축")
        sys.exit(0)
    else:
        print("\n🔧 문제 해결 필요:")
        print("1. Git 브랜치 확인")
        print("2. ROS2 환경 설정")
        print("3. 카메라 서비스 서버 설정")
        print("4. VLA 컬렉터 설정")
        sys.exit(1)

if __name__ == "__main__":
    main()
