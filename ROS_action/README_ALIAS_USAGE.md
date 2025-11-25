# 🚀 Mobile VLA Alias 사용 가이드

## 📋 추가된 Alias들

다음 alias들이 `~/.zshrc`와 `~/.bashrc`에 추가되었습니다:

### 🎯 **주요 Alias들**

1. **`vla-collect`** - 데이터 수집기 실행
   ```bash
   vla-collect
   ```
   - Mobile VLA 데이터 수집기 실행
   - 8가지 컵 도달 시나리오 지원
   - WASD 키보드 입력으로 데이터 수집

2. **`robot-move`** - 로봇 제어기 실행
   ```bash
   robot-move
   ```
   - 간단한 로봇 제어기 실행
   - WASD 키보드로 로봇 직접 조작
   - 실시간 로봇 움직임 제어

3. **`vla-system`** - 전체 시스템 실행
   ```bash
   vla-system
   ```
   - Mobile VLA 전체 시스템 실행
   - ROS2 launch 파일을 통한 시스템 시작

4. **`vla-env`** - 환경 설정
   ```bash
   vla-env
   ```
   - ROS 환경 설정 및 VLA 패키지 로드
   - 수동으로 환경을 설정할 때 사용

## 🛠️ **실행 스크립트들**

### 📁 **스크립트 위치**
- `/home/soda/vla/ROS_action/run_vla_collector.sh`
- `/home/soda/vla/ROS_action/run_robot_mover.sh`
- `/home/soda/vla/ROS_action/start_mobile_vla.sh`

### 🎮 **사용 방법**

#### 1. **데이터 수집기 실행**
```bash
# Alias 사용
vla-collect

# 또는 직접 실행
/home/soda/vla/ROS_action/run_vla_collector.sh
```

#### 2. **로봇 제어기 실행**
```bash
# Alias 사용
robot-move

# 또는 직접 실행
/home/soda/vla/ROS_action/run_robot_mover.sh
```

#### 3. **전체 시스템 실행**
```bash
# Alias 사용
vla-system

# 또는 직접 실행
/home/soda/vla/ROS_action/start_mobile_vla.sh
```

## 🎯 **조작 방법**

### **데이터 수집기 (vla-collect)**
- **N**: 새 에피소드 시작
- **1-8**: 시나리오 선택 (컵 도달 시나리오)
- **C/V**: 패턴 선택 (Core/Variant)
- **J/K/L**: 거리 선택 (Close/Medium/Far)
- **WASD**: 로봇 이동
- **M**: 에피소드 종료
- **P**: 진행 상황 확인

### **로봇 제어기 (robot-move)**
- **W**: 전진
- **A**: 왼쪽 이동
- **S**: 후진
- **D**: 오른쪽 이동
- **Q/E/Z/C**: 대각선 이동
- **R/T**: 회전
- **스페이스바**: 정지

## 🔧 **문제 해결**

### **Alias가 작동하지 않는 경우**
```bash
# zsh 사용자
source ~/.zshrc

# bash 사용자
source ~/.bashrc
```

### **환경 설정 문제**
```bash
# 수동으로 환경 설정
vla-env
```

### **직접 실행**
```bash
# 스크립트 직접 실행
cd /home/soda/vla/ROS_action
source /opt/ros/humble/setup.bash
source install/setup.bash
python3 src/mobile_vla_package/mobile_vla_package/mobile_vla_data_collector.py
```

## 📊 **현재 상태**

✅ **빌드 완료**: 모든 패키지가 성공적으로 빌드됨  
✅ **스크립트 생성**: 실행 스크립트들이 생성됨  
✅ **Alias 추가**: 편의를 위한 alias들이 추가됨  
✅ **테스트 완료**: 스크립트들이 정상적으로 실행됨  

## 🎉 **사용 준비 완료!**

이제 다음 명령어들로 Mobile VLA 시스템을 쉽게 사용할 수 있습니다:

```bash
vla-collect    # 데이터 수집 시작
robot-move     # 로봇 제어 시작
vla-system     # 전체 시스템 시작
vla-env        # 환경 설정
```
