#!/usr/bin/env python3
"""
VLA Control Center - 통합 모니터링 대시보드
학습 상태, 추론 서버 상태, 데이터셋 검증 결과를 한눈에 확인

실행: python3 scripts/control_center.py
"""

import os
import sys
import time
import json
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import subprocess

# ANSI 색상 코드
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class VLAControlCenter:
    """VLA 통합 모니터링 및 제어"""
    
    def __init__(self, project_root: str = "/home/billy/25-1kp/vla"):
        self.project_root = Path(project_root)
        self.api_url = "http://localhost:8000"
        
    def clear_screen(self):
        """화면 클리어"""
        os.system('clear' if os.name == 'posix' else 'cls')
        
    def print_header(self):
        """헤더 출력"""
        print(f"{Colors.HEADER}{Colors.BOLD}")
        print("=" * 80)
        print("🎮 VLA CONTROL CENTER".center(80))
        print("=" * 80)
        print(f"{Colors.ENDC}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
    def check_training_status(self) -> Dict:
        """학습 프로세스 상태 확인"""
        try:
            # Check for running training processes
            result = subprocess.run(
                ["ps", "aux"],
                capture_output=True,
                text=True
            )
            
            training_procs = []
            for line in result.stdout.split('\n'):
                if 'main.py' in line and 'python' in line:
                    parts = line.split()
                    training_procs.append({
                        'pid': parts[1],
                        'cpu': parts[2],
                        'mem': parts[3],
                        'time': parts[9]
                    })
            
            # Check latest log
            log_dir = self.project_root / "logs"
            if log_dir.exists():
                log_files = sorted(log_dir.glob("train_*.log"), reverse=True)
                latest_log = log_files[0] if log_files else None
                
                if latest_log:
                    # Read last few lines
                    with open(latest_log) as f:
                        lines = f.readlines()
                        last_lines = lines[-5:] if len(lines) >= 5 else lines
                        
                    # Extract progress info
                    progress_info = None
                    for line in reversed(last_lines):
                        if 'Epoch' in line and '%' in line:
                            progress_info = line.strip()
                            break
                            
                    return {
                        'running': len(training_procs) > 0,
                        'processes': training_procs,
                        'latest_log': str(latest_log),
                        'progress': progress_info
                    }
            
            return {
                'running': len(training_procs) > 0,
                'processes': training_procs,
                'latest_log': None,
                'progress': None
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def check_inference_server(self) -> Dict:
        """추론 서버 상태 확인"""
        try:
            # Health check
            response = requests.get(f"{self.api_url}/health", timeout=2)
            if response.status_code == 200:
                data = response.json()
                
                # Test prediction endpoint
                test_response = requests.get(f"{self.api_url}/", timeout=2)
                
                return {
                    'status': 'running',
                    'health': data,
                    'api_info': test_response.json()
                }
            else:
                return {'status': 'unhealthy', 'code': response.status_code}
                
        except requests.exceptions.ConnectionError:
            return {'status': 'not_running'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def check_dataset_validation(self) -> Dict:
        """데이터셋 검증 결과 확인"""
        report_path = self.project_root / "docs/dataset_validation_report.json"
        
        if report_path.exists():
            with open(report_path) as f:
                data = json.load(f)
            return {
                'available': True,
                'total': data.get('total_episodes', 0),
                'valid': data.get('valid_episodes', 0),
                'invalid': data.get('invalid_episodes', 0),
                'timestamp': data.get('timestamp', '')
            }
        else:
            return {'available': False}
    
    def print_training_status(self, status: Dict):
        """학습 상태 출력"""
        print(f"{Colors.OKBLUE}{'━' * 80}{Colors.ENDC}")
        print(f"{Colors.BOLD}📚 TRAINING STATUS{Colors.ENDC}")
        print(f"{Colors.OKBLUE}{'━' * 80}{Colors.ENDC}")
        
        if status.get('error'):
            print(f"{Colors.FAIL}❌ Error: {status['error']}{Colors.ENDC}")
            return
        
        if status['running']:
            print(f"{Colors.OKGREEN}✅ Training is RUNNING{Colors.ENDC}")
            
            for proc in status['processes']:
                print(f"\n  PID: {proc['pid']}")
                print(f"  CPU: {proc['cpu']}%  |  Memory: {proc['mem']}%  |  Time: {proc['time']}")
            
            if status['progress']:
                print(f"\n  {Colors.OKCYAN}Progress: {status['progress']}{Colors.ENDC}")
            
            if status['latest_log']:
                print(f"\n  📁 Log: {status['latest_log']}")
                print(f"     Monitor: tail -f {status['latest_log']}")
        else:
            print(f"{Colors.WARNING}⚠️  No training process running{Colors.ENDC}")
        
        print()
    
    def print_server_status(self, status: Dict):
        """서버 상태 출력"""
        print(f"{Colors.OKBLUE}{'━' * 80}{Colors.ENDC}")
        print(f"{Colors.BOLD}🚀 INFERENCE SERVER STATUS{Colors.ENDC}")
        print(f"{Colors.OKBLUE}{'━' * 80}{Colors.ENDC}")
        
        if status['status'] == 'running':
            print(f"{Colors.OKGREEN}✅ Server is RUNNING on {self.api_url}{Colors.ENDC}")
            
            health = status.get('health', {})
            print(f"\n  Status: {health.get('status', 'N/A')}")
            print(f"  Model Loaded: {health.get('model_loaded', False)}")
            print(f"  Device: {health.get('device', 'N/A')}")
            
            print(f"\n  {Colors.OKCYAN}API Endpoints:{Colors.ENDC}")
            print(f"    - GET  {self.api_url}/")
            print(f"    - GET  {self.api_url}/health")
            print(f"    - POST {self.api_url}/predict")
            print(f"    - GET  {self.api_url}/test")
            
        elif status['status'] == 'not_running':
            print(f"{Colors.WARNING}⚠️  Server is NOT RUNNING{Colors.ENDC}")
            print(f"\n  Start with: python3 Mobile_VLA/inference_server.py")
            
        else:
            print(f"{Colors.FAIL}❌ Server Error: {status.get('message', 'Unknown')}{Colors.ENDC}")
        
        print()
    
    def print_dataset_status(self, status: Dict):
        """데이터셋 검증 상태 출력"""
        print(f"{Colors.OKBLUE}{'━' * 80}{Colors.ENDC}")
        print(f"{Colors.BOLD}📊 DATASET VALIDATION{Colors.ENDC}")
        print(f"{Colors.OKBLUE}{'━' * 80}{Colors.ENDC}")
        
        if status['available']:
            total = status['total']
            valid = status['valid']
            invalid = status['invalid']
            rate = (valid / total * 100) if total > 0 else 0
            
            print(f"  Total Episodes: {total}")
            print(f"  Valid: {Colors.OKGREEN}{valid}{Colors.ENDC}  |  Invalid: {Colors.FAIL}{invalid}{Colors.ENDC}")
            print(f"  Validation Rate: {rate:.1f}%")
            print(f"\n  Last Updated: {status['timestamp']}")
            print(f"  📁 Report: docs/dataset_validation_report.md")
            
        else:
            print(f"{Colors.WARNING}⚠️  No validation report available{Colors.ENDC}")
            print(f"\n  Run: python3 scripts/validate_dataset.py")
        
        print()
    
    def print_quick_commands(self):
        """빠른 명령어 출력"""
        print(f"{Colors.OKBLUE}{'━' * 80}{Colors.ENDC}")
        print(f"{Colors.BOLD}⚡ QUICK COMMANDS{Colors.ENDC}")
        print(f"{Colors.OKBLUE}{'━' * 80}{Colors.ENDC}")
        
        commands = [
            ("🔍 모니터링", [
                "nvidia-smi",
                "tail -f logs/train_*.log",
                "watch -n 1 'ps aux | grep main.py'",
            ]),
            ("🚀 서버", [
                "python3 Mobile_VLA/inference_server.py",
                "curl http://localhost:8000/health",
                "python3 scripts/test_inference_api.py",
            ]),
            ("📚 학습", [
                "bash scripts/train_active/train_frozen_vlm.sh",
                "ps aux | grep main.py",
                "kill -9 <PID>",
            ]),
            ("🧪 테스트", [
                "python3 scripts/validate_dataset.py",
                "python3 scripts/demo_episodes.py",
            ]),
        ]
        
        for category, cmds in commands:
            print(f"\n{Colors.OKCYAN}{category}:{Colors.ENDC}")
            for cmd in cmds:
                print(f"  $ {cmd}")
        
        print()
    
    def run_dashboard(self, refresh_interval: int = 5):
        """대시보드 실행 (자동 새로고침)"""
        try:
            while True:
                self.clear_screen()
                self.print_header()
                
                # Check statuses
                training_status = self.check_training_status()
                server_status = self.check_inference_server()
                dataset_status = self.check_dataset_validation()
                
                # Print sections
                self.print_training_status(training_status)
                self.print_server_status(server_status)
                self.print_dataset_status(dataset_status)
                self.print_quick_commands()
                
                print(f"{Colors.OKBLUE}{'━' * 80}{Colors.ENDC}")
                print(f"Refreshing in {refresh_interval}s... (Press Ctrl+C to exit)")
                print(f"{Colors.OKBLUE}{'━' * 80}{Colors.ENDC}")
                
                time.sleep(refresh_interval)
                
        except KeyboardInterrupt:
            print(f"\n\n{Colors.OKGREEN}👋 Control Center closed{Colors.ENDC}\n")
    
    def run_once(self):
        """한 번만 실행"""
        self.clear_screen()
        self.print_header()
        
        training_status = self.check_training_status()
        server_status = self.check_inference_server()
        dataset_status = self.check_dataset_validation()
        
        self.print_training_status(training_status)
        self.print_server_status(server_status)
        self.print_dataset_status(dataset_status)
        self.print_quick_commands()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="VLA Control Center")
    parser.add_argument(
        "--refresh",
        type=int,
        default=5,
        help="Auto-refresh interval (seconds)"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run once without auto-refresh"
    )
    
    args = parser.parse_args()
    
    center = VLAControlCenter()
    
    if args.once:
        center.run_once()
    else:
        center.run_dashboard(refresh_interval=args.refresh)


if __name__ == "__main__":
    main()
