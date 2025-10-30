#!/usr/bin/env python3
"""
Mobile VLA 전체 구현 단계 실행
Step 1-5를 순차적으로 실행하여 완전한 Mobile VLA 시스템 구축
"""

import subprocess
import sys
import time
import logging
from pathlib import Path

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MobileVLAImplementationRunner:
    """Mobile VLA 구현 실행기"""
    
    def __init__(self):
        self.script_dir = Path(__file__).parent
        self.steps = [
            "step1_data_collection_test.py",
            "step2_mobile_vla_model.py", 
            "step3_training_pipeline.py",
            "step4_inference_system.py",
            "step5_performance_evaluation.py"
        ]
        
        self.results = {}
        
        logger.info("🚀 Mobile VLA 전체 구현 실행기 초기화")
    
    def run_step(self, step_script: str) -> bool:
        """단일 단계 실행"""
        step_path = self.script_dir / step_script
        step_name = step_script.replace(".py", "").replace("step", "Step ")
        
        logger.info(f"🔄 {step_name} 실행 시작")
        logger.info("=" * 60)
        
        try:
            # Python 스크립트 실행
            result = subprocess.run(
                [sys.executable, str(step_path)],
                capture_output=True,
                text=True,
                timeout=300  # 5분 타임아웃
            )
            
            if result.returncode == 0:
                logger.info(f"✅ {step_name} 실행 성공")
                logger.info(f"출력:\n{result.stdout}")
                self.results[step_name] = "SUCCESS"
                return True
            else:
                logger.error(f"❌ {step_name} 실행 실패")
                logger.error(f"오류:\n{result.stderr}")
                self.results[step_name] = f"FAILED: {result.stderr}"
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"⏰ {step_name} 실행 타임아웃 (5분 초과)")
            self.results[step_name] = "TIMEOUT"
            return False
        except Exception as e:
            logger.error(f"❌ {step_name} 실행 중 오류: {e}")
            self.results[step_name] = f"ERROR: {e}"
            return False
    
    def run_all_steps(self) -> bool:
        """모든 단계 순차 실행"""
        logger.info("🚀 Mobile VLA 전체 구현 시작")
        logger.info("=" * 80)
        
        success_count = 0
        total_steps = len(self.steps)
        
        for i, step_script in enumerate(self.steps, 1):
            logger.info(f"📋 진행 상황: {i}/{total_steps}")
            
            # 단계 실행
            success = self.run_step(step_script)
            
            if success:
                success_count += 1
                logger.info(f"✅ Step {i} 완료")
            else:
                logger.error(f"❌ Step {i} 실패")
                logger.error("🛑 구현 중단 - 이전 단계 문제를 해결해주세요")
                break
            
            # 단계 간 대기
            if i < total_steps:
                logger.info("⏳ 다음 단계로 진행...")
                time.sleep(2)
        
        # 결과 요약
        self.print_summary(success_count, total_steps)
        
        return success_count == total_steps
    
    def print_summary(self, success_count: int, total_steps: int):
        """결과 요약 출력"""
        logger.info("=" * 80)
        logger.info("📊 Mobile VLA 구현 결과 요약")
        logger.info("=" * 80)
        
        for step_name, result in self.results.items():
            status_icon = "✅" if result == "SUCCESS" else "❌"
            logger.info(f"{status_icon} {step_name}: {result}")
        
        logger.info("-" * 80)
        logger.info(f"📈 전체 진행률: {success_count}/{total_steps} ({success_count/total_steps*100:.1f}%)")
        
        if success_count == total_steps:
            logger.info("🎉 모든 단계 완료! Mobile VLA 시스템이 성공적으로 구축되었습니다.")
            logger.info("🎯 다음 단계:")
            logger.info("  1. 실제 데이터 수집 시작")
            logger.info("  2. 모델 학습 실행")
            logger.info("  3. Jetson에서 배포 테스트")
        else:
            logger.error("⚠️  일부 단계가 실패했습니다. 문제를 해결한 후 다시 실행해주세요.")
    
    def run_specific_step(self, step_number: int) -> bool:
        """특정 단계만 실행"""
        if step_number < 1 or step_number > len(self.steps):
            logger.error(f"❌ 잘못된 단계 번호: {step_number}")
            logger.error(f"사용 가능한 단계: 1-{len(self.steps)}")
            return False
        
        step_script = self.steps[step_number - 1]
        return self.run_step(step_script)
    
    def show_help(self):
        """도움말 출력"""
        print("""
🚀 Mobile VLA 구현 실행기

사용법:
  python run_all_steps.py [옵션]

옵션:
  --all, -a          모든 단계 순차 실행 (기본값)
  --step N, -s N     특정 단계만 실행 (N: 1-5)
  --help, -h         이 도움말 출력

단계별 설명:
  Step 1: 데이터 수집 환경 테스트
  Step 2: Mobile VLA 모델 구조 구현
  Step 3: 학습 파이프라인 구현
  Step 4: 추론 시스템 구현
  Step 5: 성능 평가 시스템 구현

예시:
  python run_all_steps.py --all
  python run_all_steps.py --step 1
  python run_all_steps.py -s 2
        """)

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Mobile VLA 구현 실행기")
    parser.add_argument("--all", "-a", action="store_true", help="모든 단계 순차 실행")
    parser.add_argument("--step", "-s", type=int, help="특정 단계만 실행 (1-5)")
    parser.add_argument("--help", "-h", action="store_true", help="도움말 출력")
    
    args = parser.parse_args()
    
    runner = MobileVLAImplementationRunner()
    
    if args.help:
        runner.show_help()
        return
    
    if args.step:
        # 특정 단계 실행
        success = runner.run_specific_step(args.step)
        sys.exit(0 if success else 1)
    else:
        # 모든 단계 실행
        success = runner.run_all_steps()
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
