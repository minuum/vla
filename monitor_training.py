import os
import time
import subprocess
import sys

def monitor():
    log_file = "/home/billy/25-1kp/vla/training_classification.log"
    pid_file = "/home/billy/25-1kp/vla/training_pid.txt"
    
    print("=" * 60)
    print("🚀 V2 Classification Training Monitoring System")
    print("=" * 60)
    
    if not os.path.exists(pid_file):
        print("❌ Training PID file not found. Is the training running?")
        return
        
    with open(pid_file, 'r') as f:
        pid = f.read().strip()
        
    print(f"✅ Training PID: {pid}")
    print(f"📂 Log File: {log_file}")
    print("-" * 60)
    
    last_size = 0
    try:
        while True:
            # 1. Check if process is still running
            if not os.path.exists(f"/proc/{pid}"):
                print("\n⚠️ Training process has stopped.")
                break
                
            # 2. Display new log lines
            if os.path.exists(log_file):
                current_size = os.path.getsize(log_file)
                if current_size > last_size:
                    with open(log_file, 'r') as f:
                        f.seek(last_size)
                        new_content = f.read()
                        if new_content:
                            sys.stdout.write(new_content)
                            sys.stdout.flush()
                        last_size = current_size
            
            # 3. Periodically show GPU status (every 30 seconds)
            # You can manualy run 'nvidia-smi' to check GPU utilization.
            
            time.sleep(2)
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped (Training continues in background).")

if __name__ == "__main__":
    monitor()
