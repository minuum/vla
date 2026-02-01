#!/usr/bin/env python3
"""
학습 모니터링 도구
학습 로그를 실시간으로 파싱하여 progress, loss, ETA 등을 표시

실행: python3 scripts/monitor_training.py --log logs/train_*.log
"""

import sys
import time
import re
from pathlib import Path
from datetime import datetime, timedelta


class TrainingMonitor:
    """학습 진행 상황 모니터링"""
    
    def __init__(self, log_file: str):
        self.log_file = Path(log_file)
        
        if not self.log_file.exists():
            raise FileNotFoundError(f"Log file not found: {log_file}")
        
        # Patterns
        self.epoch_pattern = re.compile(r'Epoch (\d+)')
        self.progress_pattern = re.compile(r'(\d+)%')
        self.loss_pattern = re.compile(r'train_loss=([\d.]+)')
        self.val_loss_pattern = re.compile(r'val_loss=([\d.]+)')
        self.iter_pattern = re.compile(r'(\d+)/(\d+)')
        
    def parse_line(self, line: str) -> dict:
        """로그 라인 파싱"""
        result = {}
        
        # Epoch
        epoch_match = self.epoch_pattern.search(line)
        if epoch_match:
            result['epoch'] = int(epoch_match.group(1))
        
        # Progress
        progress_match = self.progress_pattern.search(line)
        if progress_match:
            result['progress'] = int(progress_match.group(1))
        
        # Loss
        loss_match = self.loss_pattern.search(line)
        if loss_match:
            result['train_loss'] = float(loss_match.group(1))
        
        val_loss_match = self.val_loss_pattern.search(line)
        if val_loss_match:
            result['val_loss'] = float(val_loss_match.group(1))
        
        # Iteration
        iter_match = self.iter_pattern.search(line)
        if iter_match:
            result['current_iter'] = int(iter_match.group(1))
            result['total_iter'] = int(iter_match.group(2))
        
        return result
    
    def tail_log(self, n_lines: int = 50):
        """로그 끝부분 읽기"""
        with open(self.log_file) as f:
            lines = f.readlines()
            return lines[-n_lines:] if len(lines) >= n_lines else lines
    
    def get_latest_status(self):
        """최신 학습 상태 가져오기"""
        lines = self.tail_log(n_lines=100)
        
        latest_info = {
            'epoch': None,
            'progress': None,
            'train_loss': None,
            'val_loss': None,
            'current_iter': None,
            'total_iter': None
        }
        
        # Parse from end
        for line in reversed(lines):
            parsed = self.parse_line(line)
            
            for key, value in parsed.items():
                if latest_info[key] is None:
                    latest_info[key] = value
            
            # Stop if all found
            if all(v is not None for v in latest_info.values()):
                break
        
        return latest_info
    
    def print_status(self, status: dict):
        """상태 출력"""
        print("\n" + "="*60)
        print("📊 TRAINING STATUS".center(60))
        print("="*60)
        
        if status['epoch'] is not None:
            print(f"\n  Epoch: {status['epoch']}")
        
        if status['progress'] is not None:
            bar_length = 40
            filled = int(bar_length * status['progress'] / 100)
            bar = "█" * filled + "░" * (bar_length - filled)
            print(f"  Progress: [{bar}] {status['progress']}%")
        
        if status['current_iter'] and status['total_iter']:
            print(f"  Iteration: {status['current_iter']}/{status['total_iter']}")
            
            # ETA calculation (rough estimate)
            if status['progress'] and status['progress'] > 0:
                # Assume linear progress
                eta_percent = 100 - status['progress']
                # This is very rough, just for demo
                print(f"  Estimated remaining: ~{eta_percent}%")
        
        if status['train_loss'] is not None:
            print(f"\n  Train Loss: {status['train_loss']:.6f}")
        
        if status['val_loss'] is not None:
            print(f"  Val Loss: {status['val_loss']:.6f}")
        
        print(f"\n  Log: {self.log_file}")
        print("="*60 + "\n")
    
    def monitor_loop(self, refresh_interval: int = 5):
        """모니터링 루프"""
        try:
            while True:
                # Clear screen
                print("\033[2J\033[H")  # ANSI clear + home
                
                # Get and print status
                status = self.get_latest_status()
                self.print_status(status)
                
                print(f"Refreshing in {refresh_interval}s... (Ctrl+C to exit)")
                
                time.sleep(refresh_interval)
                
        except KeyboardInterrupt:
            print("\n👋 Monitoring stopped\n")
    
    def show_loss_history(self, n_epochs: int = 10):
        """Loss 히스토리 출력"""
        lines = self.tail_log(n_lines=10000) # Read more for history
        
        epoch_losses = {}
        
        for line in lines:
            parsed = self.parse_line(line)
            
            if 'epoch' in parsed and 'train_loss' in parsed:
                epoch = parsed['epoch']
                loss = parsed['train_loss']
                
                if epoch not in epoch_losses:
                    epoch_losses[epoch] = []
                
                epoch_losses[epoch].append(loss)
        
        # Print summary
        print("\n" + "="*60)
        print("📈 LOSS HISTORY".center(60))
        print("="*60 + "\n")
        
        sorted_epochs = sorted(epoch_losses.keys())[-n_epochs:]
        
        for epoch in sorted_epochs:
            losses = epoch_losses[epoch]
            avg_loss = sum(losses) / len(losses)
            min_loss = min(losses)
            max_loss = max(losses)
            
            print(f"  Epoch {epoch}:")
            print(f"    Avg Loss: {avg_loss:.6f}")
            print(f"    Min Loss: {min_loss:.6f}")
            print(f"    Max Loss: {max_loss:.6f}")
            print(f"    Samples: {len(losses)}")
            print()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor training")
    parser.add_argument(
        "--log",
        type=str,
        required=True,
        help="Path to training log file"
    )
    parser.add_argument(
        "--refresh",
        type=int,
        default=5,
        help="Refresh interval (seconds)"
    )
    parser.add_argument(
        "--history",
        action="store_true",
        help="Show loss history instead of live monitoring"
    )
    
    args = parser.parse_args()
    
    monitor = TrainingMonitor(log_file=args.log)
    
    if args.history:
        monitor.show_loss_history()
    else:
        monitor.monitor_loop(refresh_interval=args.refresh)


if __name__ == "__main__":
    main()
