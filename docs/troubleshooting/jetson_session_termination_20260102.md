# Jetson 세션 종료 원인 분석

**일시**: 2026-01-02 01:00 KST  
**호스트**: healthinesss (Jetson Orin)

## 📊 현재 상태

### 시스템 정보
- **Uptime**: 1일 12시간 45분 (2025년 6월 4일부터 재부팅 없음)
- **메모리**: 15GB 총, 4.5GB 사용, 10GB 여유
- **디스크**: 233GB 총, 155GB 사용 (71%)
- **GPU**: Orin (nvgpu), 실행 중인 프로세스 없음

### 실행 중이던 프로세스 (종료됨)
ADDITIONAL_METADATA 기록 기준:
1. `colcon list` - 5분 44초 실행 중이었음
2. `export PYTHONPATH=...` - 4분 5초 실행 중이었음

### 현재 프로세스 상태
- **API 서버**: 실행 안됨 (uvicorn 없음)
- **Python 프로세스**: 시스템 프로세스만 존재
- **ROS2 프로세스**: 없음

## 🔍 분석 결과

### 1. 메모리 상태
- 현재 메모리 충분 (10GB 여유)
- OOM killer 관련 로그 발견:
  - `systemd[1]: Condition check resulted in Userspace Out-Of-Memory (OOM) Killer being skipped.`
  - 하지만 실제 OOM kill 로그는 없음 (sudo 권한 없어서 전체 확인 불가)

### 2. 프로세스 종료 패턴
- 모든 사용자 프로세스가 종료됨
- 시스템 레벨 프로세스만 실행 중
- **가능성**: 
  - SSH 세션 종료로 인한 프로세스 종료
  - 터미널 접속이 끊어지면서 실행 중이던 명령들도 함께 종료

### 3. 최근 활동 (zsh history)
```
python3 /tmp/analyze_docs.py
python3 /tmp/reorganize_docs.py  
python3 /tmp/execute_reorganize.py --execute
python3 /tmp/merge_subfolders.py --execute
.venv/bin/python test_local_inference_engine.py
```

### 4. ROS2 로그
- 마지막 로그: 2026-01-02 00:08:42
- `colcon list` 명령 실행 기록

## 💡 결론

### 주요 원인: **SSH 세션 종료**

1. **네트워크 불안정** 또는 **SSH 타임아웃**로 원격 세션이 끊어짐
2. 백그라운드로 실행하지 않은 프로세스들이 세션과 함께 종료
3. `nohup` 또는 `screen`/`tmux` 없이 실행한 명령들이 모두 종료됨

### 증거
- ✅ 시스템 재부팅 없음 (uptime 1일 이상)
- ✅ 메모리 충분 (OOM kill 아님)
- ✅ 디스크 공간 충분
- ✅ 사용자 프로세스만 종료 (시스템 프로세스는 정상)
- ✅ 실행 시간이 길었던 명령들 (5분+)

### OOM이 아닌 이유
- 현재 메모리 10GB 여유
- dmesg에 실제 OOM kill 로그 없음
- 시스템 재부팅 없음
- GPU 프로세스 없음 (메모리 부족 상태 아님)

## 🎯 해결 방안

### 1. 즉시 적용 (필수)
**장시간 실행 명령은 백그라운드로**:
```bash
# nohup 사용
nohup command > logs/output.log 2>&1 &

# screen 사용 (선호)
screen -S session_name
# 명령 실행
# Ctrl+A, D로 detach

# tmux 사용
tmux new -s session_name
# 명령 실행  
# Ctrl+B, D로 detach
```

### 2. API 서버 재시작
```bash
cd /home/soda/vla
export VLA_API_KEY="$(cat secrets.sh | grep VLA_API_KEY | cut -d'=' -f2 | tr -d '"' 2>/dev/null || echo 'your-key')"

# Background로 실행 (필수!)
nohup poetry run uvicorn Mobile_VLA.inference_server:app \
  --host 0.0.0.0 --port 8000 \
  > logs/api_server_$(date +%Y%m%d_%H%M%S).log 2>&1 &

echo $! > logs/server.pid
```

### 3. ROS2 환경 재설정
```bash
# screen/tmux 세션에서 실행
screen -S ros2_env

source /opt/ros/humble/setup.zsh
cd /home/soda/vla/ROS_action
colcon build
source install/setup.zsh

# Python path 설정
export PYTHONPATH=/home/soda/vla:/home/soda/vla/RoboVLMs:/home/soda/vla/RoboVLMs/RoboVLMs:$PYTHONPATH
```

### 4. SSH 세션 관리
```bash
# SSH config에 keepalive 설정
# ~/.ssh/config
Host jetson
    HostName healthinesss
    ServerAliveInterval 60
    ServerAliveCountMax 3
    TCPKeepAlive yes
```

### 5. systemd 서비스 생성 (장기 해결)
API 서버를 systemd 서비스로 등록하여 자동 재시작:
```bash
# /etc/systemd/system/vla-api.service
[Unit]
Description=VLA API Server
After=network.target

[Service]
Type=simple
User=soda
WorkingDirectory=/home/soda/vla
Environment="VLA_API_KEY=your-key"
ExecStart=/home/soda/vla/.venv/bin/uvicorn Mobile_VLA.inference_server:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

## 📋 체크리스트

다음부터 장시간 실행 작업 시:
- [ ] `screen` 또는 `tmux` 세션 사용
- [ ] `nohup`으로 백그라운드 실행
- [ ] 로그 파일 지정 (`> logs/output.log 2>&1`)
- [ ] PID 파일 저장 (`echo $! > logs/process.pid`)
- [ ] SSH keepalive 설정 확인

## 📝 참고
- 이전 작업: docs 리팩토링 완료 (SUMMARY_20260101.md)
- 다음 작업: Phase 2 로컬 추론 노드 테스트
- 관련 문서: `docs/inference/20251224_remaining_tasks.md`
