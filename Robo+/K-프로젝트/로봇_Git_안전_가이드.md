# 🛡️ **로봇 환경 Git 안전 사용 가이드 - 파일 손실 완전 방지**

> 💡 **목적**: 로봇 환경에서 git pull 시 중요 파일이 삭제되는 문제를 완전히 방지하고, 안전한 Git 작업 환경 구축

---

## 🚨 **문제 상황 분석**

### **💔 왜 파일이 지워지는가?**
```bash
# 위험한 상황들
git pull origin main          # ❌ 충돌 시 파일 손실 위험
git reset --hard              # ❌ 모든 로컬 변경사항 삭제
git checkout -- .             # ❌ 작업 중인 파일들 삭제
git clean -fd                 # ❌ 추적되지 않는 파일들 삭제
```

### **🔍 실제 발생했던 문제들**
1. **Recent Changes 에서 확인된 문제**:
   - `git reset --hard origin/feature-action`으로 K-프로젝트 핵심 파일들 손실
   - 서브모듈 해제 과정에서 일부 스크립트 손실
   - GitHub Push Protection으로 인한 히스토리 조작 필요

2. **로봇 환경 특수성**:
   - 네트워크 불안정으로 인한 부분 pull
   - 실시간 시스템 실행 중 Git 작업으로 인한 충돌
   - 백업 시스템 부재로 인한 복구 불가

---

## 🛡️ **완전한 해결책**

### **🎯 1단계: 안전한 Git Pull 스크립트 사용**

#### **기본 사용법**
```bash
# 로봇에서 항상 이것만 사용하세요!
cd vla/RoboVLMs
./git_safe_pull.sh

# 강제 pull이 필요한 경우 (신중히!)
./git_safe_pull.sh --force

# 백업만 수행
./git_safe_pull.sh --backup-only

# 백업에서 복구
./git_safe_pull.sh --restore
```

#### **안전 스크립트의 보호 기능**
```bash
✅ VLA 시스템 자동 종료
✅ 중요 파일 자동 백업
✅ Git 상태 사전 확인
✅ 충돌 사전 감지
✅ 실패 시 자동 복구 옵션
✅ 백업 이력 관리
```

### **🎯 2단계: 중요 파일 자동 보호**

#### **보호되는 핵심 파일들**
```bash
# 실행 스크립트들
jetson_quick_start.sh
launch_event_triggered_vla.sh
send_text_command.sh
stop_event_triggered_vla.sh
git_safe_pull.sh

# 설정 파일들
docker-compose.yml
configs/k_project/ros2_automotive.json

# 문서 및 가이드
../Robo+/K-프로젝트/
../Model_ws/src/vla_node/

# 모델 캐시 및 데이터
models_cache/
.vlms/
```

#### **자동 백업 시스템**
```bash
# 백업 디렉토리 구조
backup_20250725_143022/
├── backup_info.txt              # 백업 정보
├── jetson_quick_start.sh        # 실행 스크립트들
├── launch_event_triggered_vla.sh
├── configs/k_project/           # 설정 파일들
└── ../Robo+/K-프로젝트/          # 문서들
```

### **🎯 3단계: Git 설정 최적화**

#### **로봇용 안전 Git 설정**
```bash
# 한 번만 설정하면 됩니다
git config --global pull.rebase false
git config --global merge.ours.driver true
git config --global core.autocrlf false
git config --global push.default simple

# 자동 stash 설정 (변경사항 자동 보관)
git config --global rebase.autostash true

# 대용량 파일 경고 설정
git config --global core.bigFileThreshold 100m
```

#### **안전한 .gitignore 설정**
```bash
# 로봇 환경 전용 .gitignore 추가
echo "
# 로봇 환경 보호 파일들
backup_*/
*.backup
*.bak
models_cache/
.vlms/
*.log
core.*
.pytest_cache/
__pycache__/
*.pyc
.venv/
venv/
" >> .gitignore
```

---

## 🚀 **실전 사용법**

### **📋 일상적인 Git 작업 순서**

#### **1. 작업 시작 전 (매번 실행)**
```bash
# 1. VLA 시스템 종료 확인
docker ps | grep k_project_event_vla
# 실행 중이면: ./stop_event_triggered_vla.sh

# 2. 백업 수행
./git_safe_pull.sh --backup-only

# 3. 안전한 pull
./git_safe_pull.sh

# 4. 권한 확인 및 테스트
chmod +x *.sh
./jetson_quick_start.sh
```

#### **2. 작업 중 변경사항 관리**
```bash
# 중요 변경사항은 즉시 커밋
git add 중요한_파일.sh
git commit -m "로봇: 중요 기능 추가"

# 임시 변경사항은 stash
git stash push -m "작업 중 임시 저장"

# 주기적 백업 (하루 1회)
./git_safe_pull.sh --backup-only
```

#### **3. 문제 발생 시 대응**
```bash
# 즉시 복구
./git_safe_pull.sh --restore

# 특정 파일만 복구
cp backup_최신/파일이름 ./

# Git 상태 완전 초기화 (최후 수단)
git stash
git reset --hard HEAD
git clean -fd
./git_safe_pull.sh --restore
```

---

## 🔧 **고급 보호 기능**

### **🛡️ 자동 백업 시스템 구축**

#### **cron을 이용한 자동 백업**
```bash
# crontab 설정 (매일 새벽 3시 백업)
crontab -e

# 추가할 내용:
0 3 * * * cd /home/robot/vla/RoboVLMs && ./git_safe_pull.sh --backup-only >> /var/log/git_backup.log 2>&1
```

#### **systemd 서비스를 이용한 자동 보호**
```bash
# /etc/systemd/system/git-safety.service
[Unit]
Description=Git Safety Monitor for K-Project
After=network.target

[Service]
Type=simple
User=robot
WorkingDirectory=/home/robot/vla/RoboVLMs
ExecStart=/home/robot/vla/RoboVLMs/git_safe_pull.sh --backup-only
Restart=always
RestartSec=3600

[Install]
WantedBy=multi-user.target

# 서비스 활성화
sudo systemctl enable git-safety.service
sudo systemctl start git-safety.service
```

### **🔍 실시간 파일 모니터링**

#### **inotify를 이용한 파일 변경 감지**
```bash
#!/bin/bash
# file_monitor.sh - 중요 파일 변경 감지

inotifywait -m -r --format '%T %w %f %e' --timefmt '%Y-%m-%d %H:%M:%S' \
    -e delete,move,modify \
    jetson_quick_start.sh \
    launch_event_triggered_vla.sh \
    send_text_command.sh \
    configs/ \
    | while read timestamp path file event; do
        echo "[$timestamp] $event: $path$file"
        
        # 중요 파일 삭제 감지 시 즉시 복구
        if [[ $event == "DELETE" ]]; then
            echo "🚨 중요 파일 삭제 감지! 자동 복구 중..."
            ./git_safe_pull.sh --restore
        fi
    done
```

---

## 🎯 **로봇별 맞춤 설정**

### **🤖 Jetson 환경**
```bash
# Jetson 전용 설정
export JETSON_MODEL="orin_nx_16gb"
export GIT_BACKUP_PATH="/media/usb/git_backups"
export MAX_BACKUP_COUNT=10

# 메모리 부족 시 Git 작업 제한
free -m | awk 'NR==2{printf "%.1f%%", $3*100/$2}'
if [ $(free -m | awk 'NR==2{printf "%.0f", $3*100/$2}') -gt 80 ]; then
    echo "⚠️ 메모리 부족. Git 작업을 연기하세요."
fi
```

### **🔌 네트워크 불안정 환경**
```bash
# 네트워크 상태 확인 함수
check_network() {
    if ! ping -c 1 github.com > /dev/null 2>&1; then
        echo "❌ 네트워크 연결 불안정. Git 작업을 연기하세요."
        return 1
    fi
    return 0
}

# timeout을 이용한 안전한 pull
timeout 60 git pull origin main || {
    echo "⚠️ Pull 시간 초과. 네트워크 상태를 확인하세요."
    exit 1
}
```

---

## 📊 **모니터링 및 알림**

### **🔔 Slack/Discord 알림 설정**
```bash
# Webhook을 이용한 알림 함수
notify_team() {
    local message="$1"
    local webhook_url="YOUR_WEBHOOK_URL"
    
    curl -X POST -H 'Content-type: application/json' \
        --data "{\"text\":\"🤖 로봇 Git 알림: $message\"}" \
        "$webhook_url"
}

# 사용 예시
notify_team "백업 완료: $(date)"
notify_team "⚠️ Git pull 실패 - 복구 필요"
```

### **📈 백업 상태 대시보드**
```bash
# backup_status.sh - 백업 상태 확인
#!/bin/bash

echo "=== K-프로젝트 백업 상태 ==="
echo "현재 시간: $(date)"
echo ""

# 백업 디렉토리 현황
echo "📦 백업 현황:"
ls -la backup_* 2>/dev/null | head -10 || echo "백업이 없습니다"

echo ""
echo "💾 디스크 사용량:"
df -h . | tail -1

echo ""
echo "🔒 중요 파일 상태:"
for file in jetson_quick_start.sh launch_event_triggered_vla.sh; do
    if [ -f "$file" ]; then
        echo "✅ $file ($(stat -c%s $file) bytes)"
    else
        echo "❌ $file (누락!)"
    fi
done
```

---

## 🆘 **비상 복구 절차**

### **🚨 모든 파일이 사라진 경우**
```bash
# 1단계: 패닉하지 말고 확인
ls -la
git status
docker ps

# 2단계: 백업에서 복구 시도
./git_safe_pull.sh --restore

# 3단계: 백업도 없는 최악의 상황
# GitHub에서 직접 다운로드
wget https://github.com/minuum/vla/archive/feature-action.zip
unzip feature-action.zip
cp -r vla-feature-action/* ./

# 4단계: 권한 복구
chmod +x *.sh
```

### **🔧 부분 파일 손실 시**
```bash
# 특정 파일만 복구
git checkout HEAD -- 파일이름

# 또는 GitHub에서 직접 다운로드
curl -O https://raw.githubusercontent.com/minuum/vla/feature-action/RoboVLMs/jetson_quick_start.sh
chmod +x jetson_quick_start.sh
```

---

## 📚 **Best Practices**

### **✅ 해야 할 것들**
1. **항상 안전 스크립트 사용**: `./git_safe_pull.sh`
2. **작업 전 VLA 시스템 종료**: `./stop_event_triggered_vla.sh`
3. **중요 변경사항 즉시 커밋**: `git add . && git commit -m "중요 변경"`
4. **주기적 백업**: 하루 1회 이상
5. **네트워크 상태 확인**: ping 테스트 후 Git 작업
6. **권한 확인**: `chmod +x *.sh` 항상 실행

### **❌ 하지 말아야 할 것들**
1. **절대 사용 금지**: `git reset --hard`, `git clean -fd`
2. **VLA 실행 중 Git 작업**: 시스템 충돌 위험
3. **백업 없는 강제 pull**: 파일 손실 위험
4. **네트워크 불안정 시 Git 작업**: 부분 다운로드 위험
5. **권한 확인 생략**: 스크립트 실행 불가

---

## 🎯 **요약: 핵심 명령어 치트시트**

```bash
# 🛡️ 매일 사용하는 안전한 Git 명령어들
./stop_event_triggered_vla.sh           # 1. 시스템 종료
./git_safe_pull.sh                      # 2. 안전한 업데이트  
chmod +x *.sh                           # 3. 권한 확인
./jetson_quick_start.sh                 # 4. 환경 테스트
./launch_event_triggered_vla.sh         # 5. 시스템 시작

# 🆘 문제 발생 시 복구 명령어들
./git_safe_pull.sh --restore            # 백업에서 복구
docker ps | grep k_project              # 시스템 상태 확인
git status                              # Git 상태 확인
```

---

**💡 핵심**: 로봇 환경에서는 **항상 안전을 최우선**으로! 의심스러우면 백업부터, 확실하지 않으면 팀에게 문의하세요.

**📞 긴급 상황**: 복구 불가능한 파일 손실 시 즉시 팀원들에게 알리고, 이 가이드의 비상 복구 절차를 따르세요.

**📅 마지막 업데이트**: 2025년 7월 25일  
**👥 작성자**: K-프로젝트 팀 Git 안전 위원회