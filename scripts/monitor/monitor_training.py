#!/usr/bin/env python3
"""
Mobile-VLA 학습 모니터링 도구
선택 메뉴: 모니터링 스크립트 / Tensorboard / 실시간 로그
"""

import os
import sys
import glob
import subprocess
from pathlib import Path


class TrainingMonitor:
    def __init__(self, experiment_name="mobile_vla_kosmos2_frozen_lora_leftright_20251204"):
        self.experiment_name = experiment_name
        self.run_dir = f"RoboVLMs_upstream/runs/{experiment_name}"
        self.log_files = sorted(glob.glob("case3_kosmos2_leftright_*.txt"))
        self.log_file = self.log_files[-1] if self.log_files else None
        
    def show_menu(self):
        """메뉴 출력"""
        print("\n" + "="*60)
        print("📊 Mobile-VLA 학습 모니터링 도구")
        print("="*60)
        print(f"실험: {self.experiment_name}")
        print(f"로그: {self.log_file if self.log_file else '없음'}")
        print()
        print("선택 가능한 옵션:")
        print("  [1] 📊 모니터링 요약 보기")
        print("  [2] 📈 Tensorboard 실행")
        print("  [3] 📜 실시간 로그 (tail -f)")
        print("  [4] 🔄 새로고침")
        print("  [0] 종료")
        print("="*60)
        
    def check_process(self):
        """프로세스 상태 확인"""
        try:
            result = subprocess.run(
                f"ps aux | grep 'python.*main.py.*{self.experiment_name}' | grep -v grep",
                shell=True, capture_output=True, text=True
            )
            if result.stdout.strip():
                pid = result.stdout.split()[1]
                return True, pid
            return False, None
        except:
            return False, None
    
    def get_checkpoints(self):
        """Checkpoint 정보"""
        if not Path(self.run_dir).exists():
            return []
        
        ckpts = list(Path(self.run_dir).rglob("*.ckpt"))
        ckpts.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return ckpts[:5]  # 최근 5개
    
    def show_summary(self):
        """모니터링 요약"""
        print("\n" + "="*60)
        print("📊 모니터링 요약")
        print("="*60)
        
        # 프로세스 상태
        is_running, pid = self.check_process()
        print("\n[1] 프로세스 상태")
        print("-"*60)
        if is_running:
            print(f"  ✅ 실행 중 (PID: {pid})")
        else:
            print("  ❌ 실행 안 됨")
        
        # 최근 로그
        if self.log_file and Path(self.log_file).exists():
            print("\n[2] 최근 로그 (10 lines)")
            print("-"*60)
            result = subprocess.run(
                f"tail -10 {self.log_file} | grep -E 'Epoch|Loss|Error|training|validation' || tail -10 {self.log_file}",
                shell=True, capture_output=True, text=True
            )
            print(result.stdout)
        
        # Checkpoint
        ckpts = self.get_checkpoints()
        print("[3] Checkpoint 현황")
        print("-"*60)
        print(f"  저장된 checkpoint: {len(ckpts)}개")
        for i, ckpt in enumerate(ckpts[:3], 1):
            size = ckpt.stat().st_size / (1024**3)
            print(f"  {i}. {ckpt.name} ({size:.1f}GB)")
        
        print()
        input("Press Enter to continue...")
    
    def run_tensorboard(self):
        """Tensorboard 실행"""
        print("\n" + "="*60)
        print("📈 Tensorboard 실행")
        print("="*60)
        
        if not Path(self.run_dir).exists():
            print("  ❌ Run 디렉토리 없음")
            input("Press Enter to continue...")
            return
        
        print(f"  Starting Tensorboard...")
        print(f"  URL: http://localhost:6006")
        print()
        print("  종료: Ctrl+C")
        print()
        
        try:
            subprocess.run(
                f"tensorboard --logdir {self.run_dir}",
                shell=True
            )
        except KeyboardInterrupt:
            print("\n  Tensorboard 종료됨")
    
    def tail_log(self):
        """실시간 로그"""
        if not self.log_file or not Path(self.log_file).exists():
            print("\n  ❌ 로그 파일 없음")
            input("Press Enter to continue...")
            return
        
        print("\n" + "="*60)
        print(f"📜 실시간 로그: {self.log_file}")
        print("="*60)
        print("  종료: Ctrl+C")
        print()
        
        try:
            subprocess.run(f"tail -f {self.log_file}", shell=True)
        except KeyboardInterrupt:
            print("\n  로그 보기 종료됨")
    
    def run(self):
        """메인 루프"""
        while True:
            os.system('clear' if os.name != 'nt' else 'cls')
            self.show_menu()
            
            try:
                choice = input("\n선택 (0-4): ").strip()
                
                if choice == '0':
                    print("\n종료합니다.")
                    break
                elif choice == '1':
                    self.show_summary()
                elif choice == '2':
                    self.run_tensorboard()
                elif choice == '3':
                    self.tail_log()
                elif choice == '4':
                    continue
                else:
                    print("\n잘못된 선택입니다.")
                    input("Press Enter to continue...")
            
            except KeyboardInterrupt:
                print("\n\n종료합니다.")
                break
            except Exception as e:
                print(f"\n오류: {e}")
                input("Press Enter to continue...")


def main():
    """메인 함수"""
    # 실험명 인자로 받기 (선택)
    experiment = "mobile_vla_kosmos2_frozen_lora_leftright_20251204"
    if len(sys.argv) > 1:
        experiment = sys.argv[1]
    
    monitor = TrainingMonitor(experiment)
    monitor.run()


if __name__ == "__main__":
    main()
