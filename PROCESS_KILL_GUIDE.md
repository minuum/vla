# 프로세스 종료 가이드

## 1. 프로세스 확인 방법

### 방법 1: ps와 grep 사용
```bash
ps aux | grep -E "(camera_service_server|run_camera_server|vla_collector|run_vla_collector)" | grep -v grep
```

### 방법 2: pgrep 사용
```bash
pgrep -f "camera_service_server|run_camera_server|vla_collector|run_vla_collector"
```

### 방법 3: htop/top 사용
```bash
htop
# 또는
top
# F4로 필터링하여 프로세스 이름으로 검색
```

## 2. 프로세스 종료 방법

### 방법 1: kill 명령어 사용 (PID 직접 지정)
```bash
# 프로세스 ID 확인
pgrep -f "camera_service_server"
pgrep -f "vla_collector"

# 종료 (SIGTERM - 정상 종료)
kill <PID>

# 강제 종료 (SIGKILL - 즉시 종료, 권장하지 않음)
kill -9 <PID>
```

### 방법 2: pkill 사용 (프로세스 이름으로)
```bash
# 정상 종료
pkill -f "camera_service_server"
pkill -f "run_camera_server"
pkill -f "vla_collector"
pkill -f "run_vla_collector"

# 강제 종료
pkill -9 -f "camera_service_server"
```

### 방법 3: killall 사용
```bash
killall camera_service_server
killall -9 camera_service_server  # 강제 종료
```

### 방법 4: 한 번에 모두 종료
```bash
# 모든 관련 프로세스 종료
pkill -f "camera_service_server|run_camera_server|vla_collector|run_vla_collector"

# 또는
kill $(pgrep -f "camera_service_server|run_camera_server|vla_collector|run_vla_collector")
```

## 3. ROS2 관련 프로세스 종료

### ROS2 노드 확인
```bash
ros2 node list
```

### ROS2 노드 종료
```bash
# 특정 노드 종료
ros2 node kill <node_name>

# 모든 노드 종료
ros2 daemon stop
```

## 4. 종료 후 확인

```bash
# 프로세스가 정말 종료되었는지 확인
ps aux | grep -E "(camera_service_server|run_camera_server|vla_collector|run_vla_collector)" | grep -v grep

# 아무것도 출력되지 않으면 종료 완료
```

## 5. 문제 해결: rclpy shutdown 에러

Ctrl+C로 종료할 때 `rcl_shutdown already called` 에러가 발생하는 경우:

1. **정상 종료 방법 사용**
   ```bash
   pkill -f "camera_service_server"
   ```

2. **프로세스가 완전히 종료될 때까지 대기**
   ```bash
   sleep 2
   ps aux | grep camera_service_server | grep -v grep
   ```

3. **강제 종료 (최후의 수단)**
   ```bash
   pkill -9 -f "camera_service_server"
   ```

## 6. 빠른 종료 스크립트

```bash
#!/bin/bash
# kill_ros_processes.sh

echo "ROS 관련 프로세스 종료 중..."

# 카메라 서버 종료
pkill -f "camera_service_server"
pkill -f "run_camera_server"

# VLA 컬렉터 종료
pkill -f "vla_collector"
pkill -f "run_vla_collector"

# 확인
sleep 1
echo "남은 프로세스:"
ps aux | grep -E "(camera_service_server|run_camera_server|vla_collector|run_vla_collector)" | grep -v grep

echo "완료!"
```

## 7. 권장 사항

1. **정상 종료 우선**: `kill` 또는 `pkill` (SIGTERM) 먼저 시도
2. **강제 종료는 최후의 수단**: `kill -9` 또는 `pkill -9`는 데이터 손실 가능
3. **종료 후 확인**: 프로세스가 완전히 종료되었는지 확인
4. **ROS2 데몬 정리**: 필요시 `ros2 daemon stop` 실행



