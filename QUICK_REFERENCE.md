# VLA 빠른 참조 가이드

## 🏗️ 서버 구조

```
Jetson (fevers)          ←→          Billy (A5000)
로봇 서버                Tailscale     모델 서버
100.85.118.58                         100.XXX.XXX.XXX

• 이미지 캡쳐                          • VLA 모델 추론
• 로봇 주행                            • API 서버
• ROS2 노드                           • GPU 연산
```

## 🚀 빠른 시작

### Jetson Server (Robot)
```bash
source ~/.zshrc          # Aliases 로드
vla-jetson-env           # ROS2 환경 설정
vla-network-check        # Billy 연결 확인
vla-collect              # 데이터 수집 시작
```

### Billy Server (Model)
```bash
source ~/.zshrc          # Aliases 로드
vla-model-env            # 모델 환경 설정
vla-start                # API 서버 시작
vla-status               # 상태 확인
```

## 📋 필수 명령어

### 공통
| 명령어 | 설명 |
|--------|------|
| `vla-env` | 환경 변수 확인 |
| `vla-network-check` | 네트워크 테스트 |
| `vla-help` | 도움말 |

### Jetson 전용
| 명령어 | 설명 |
|--------|------|
| `vla-jetson-env` | ROS2 환경 설정 |
| `vla-curl-health` | Billy API Health check |

### Billy 전용
| 명령어 | 설명 |
|--------|------|
| `vla-model-env` | 모델 환경 설정 |
| `vla-start` | API 서버 시작 |
| `vla-status` | API 서버 상태 |

## 🔧 설정

### 1. Tailscale IP 확인
```bash
tailscale ip -4
```

### 2. Billy IP 설정 (Jetson에서)
```bash
# ~/.zshrc에 추가
export BILLY_TAILSCALE_IP="100.XXX.XXX.XXX"
```

### 3. API Key 설정 (Jetson에서)
```bash
# Billy에서 생성된 키 확인
vla-logs | grep "API Key"

# Jetson에서 설정
export VLA_API_KEY="<generated-key>"
```

## 🔍 문제 해결

### 연결 안 됨
```bash
vla-network-check        # 연결 테스트
ping $BILLY_TAILSCALE_IP # Ping 테스트
```

### API 서버 문제
```bash
vla-status               # 상태 확인
vla-logs                 # 로그 확인
vla-restart              # 재시작
```

### ROS2 문제
```bash
vla-jetson-env           # 환경 재설정
```

## 📖 상세 문서

- 전체 설정: `docs/VLA_MULTI_SERVER_SETUP.md`
- API 서버: `BILLY_SERVER_START_GUIDE.md`
- 빠른 시작: `QUICKSTART.md`

---

**Jetson**: `vla-jetson-env` → `vla-collect`  
**Billy**: `vla-model-env` → `vla-start`
