# VLA 멀티 서버 환경 재구성 완료 ✅

## 📅 작성: 2025-12-17 19:15

## 🎯 변경 사항 요약

기존의 단일 서버 구조에서 **Jetson-Billy 분리 구조**로 완전히 재구성되었습니다.

### Before (단일 서버)
```
vla-env          → API 환경 변수 또는 ROS 환경 (혼동)
vla-ros-env      → ROS 환경
```

### After (멀티 서버)
```
vla-jetson-env   → Jetson 서버 환경 (ROS2, 로봇 제어)
vla-model-env    → Model 서버 환경 (VLA 추론)
vla-env          → 환경 변수 확인 (공통)
```

## 🏗️ 새로운 시스템 구조

```
┌─────────────────────┐         ┌─────────────────────┐
│  Jetson (fevers)    │◄───────►│  Billy (A5000)      │
│  로봇 서버           │Tailscale│  모델 서버           │
├─────────────────────┤   VPN   ├─────────────────────┤
│ • 이미지 캡쳐        │         │ • VLA 추론          │
│ • 로봇 주행          │         │ • API 서버          │
│ • ROS2 노드          │         │ • GPU 연산          │
│ • 데이터 수집        │         │                     │
│                     │         │                     │
│ 100.85.118.58       │         │ 100.XXX.XXX.XXX     │
└─────────────────────┘         └─────────────────────┘
```

## 📦 생성된 파일

### 1. **`.vla_aliases`** (완전 재작성)
- 서버 역할 자동 감지 (hostname 기반)
- 서버별 맞춤 명령어
- Tailscale 통신 설정
- 네트워크 테스트 기능

### 2. **`docs/VLA_MULTI_SERVER_SETUP.md`** (신규)
- 전체 시스템 아키텍처
- 양쪽 서버 설정 가이드
- 통신 흐름 설명
- 상세한 트러블슈팅

### 3. **`QUICK_REFERENCE.md`** (신규)
- 빠른 참조 카드
- 자주 사용하는 명령어
- 필수 설정 요약

### 4. **`BILLY_SETUP.md`** (신규)
- Billy 서버 담당자용 가이드
- 처음부터 끝까지 단계별 설명
- API Key 생성 및 공유 방법

## 🔧 주요 기능

### 서버 자동 감지
```bash
source ~/.zshrc
# Jetson에서: "✓ VLA aliases loaded - Jetson Robot Server"
# Billy에서: "✓ VLA aliases loaded - Billy Model Server"
```

### 서버별 맞춤 명령어

#### Jetson Server
```bash
vla-jetson-env       # ROS2 환경 설정
vla-collect          # 데이터 수집
robot-move           # 로봇 이동
vla-system           # VLA 시스템
vla-network-check    # Billy 연결 확인
```

#### Billy Server
```bash
vla-model-env        # 모델 환경 설정
vla-start            # API 서버 시작
vla-status           # 서버 상태
vla-logs             # 서버 로그
vla-network-check    # Jetson 연결 확인
```

#### 공통
```bash
vla-env              # 환경 변수 확인
vla-help             # 도움말 (서버별로 다른 내용)
vla-health           # API Health check
vla-test             # API 테스트
vla-gpu              # GPU 상태 (tegrastats/nvidia-smi)
```

## 🚀 빠른 시작

### Jetson Server에서
```bash
source ~/.zshrc
vla-jetson-env
vla-network-check    # Billy 연결 확인
vla-curl-health      # Billy API 확인
```

### Billy Server에서
```bash
source ~/.zshrc
vla-model-env
vla-start
vla-status
```

## 📋 설정 체크리스트

### Jetson Server
- [x] `.vla_aliases` 재작성
- [ ] Billy Tailscale IP 설정 필요
- [ ] Billy API Key 설정 필요
- [x] `vla-jetson-env` 테스트 완료
- [x] 서버 역할 자동 감지 확인

### Billy Server (할 일)
- [ ] `.vla_aliases` 파일 복사
- [ ] Tailscale IP 확인 및 설정
- [ ] API 서버 시작
- [ ] API Key 생성 및 Jetson에 전달

## 🔑 필수 설정 항목

### Jetson Server (`~/.zshrc`)
```bash
export BILLY_TAILSCALE_IP="100.XXX.XXX.XXX"  # Billy 서버 IP
export VLA_API_KEY="<billy-api-key>"          # Billy에서 생성된 키
```

### Billy Server (`~/.zshrc`)
```bash
export BILLY_TAILSCALE_IP="<자신의-IP>"       # 자신의 Tailscale IP
```

## 📊 명령어 비교표

| 목적 | Jetson | Billy |
|------|--------|-------|
| 환경 설정 | `vla-jetson-env` | `vla-model-env` |
| 서버 시작 | - | `vla-start` |
| 연결 확인 | `vla-network-check` | `vla-network-check` |
| API 테스트 | `vla-curl-health` | `vla-test` |
| GPU 상태 | `vla-gpu` (tegrastats) | `vla-gpu` (nvidia-smi) |

## 📖 문서 구조

```
/home/soda/vla/
├── .vla_aliases                       # ⭐ 핵심 aliases (재작성)
├── QUICK_REFERENCE.md                 # 빠른 참조
├── BILLY_SETUP.md                     # Billy 서버 설정 가이드
├── docs/
│   └── VLA_MULTI_SERVER_SETUP.md     # 종합 설정 가이드
├── scripts/
│   ├── manage_api_server.sh          # API 서버 관리
│   ├── test_api_server.sh            # API 테스트
│   └── README.md                      # 스크립트 문서
└── (기존 파일들...)
```

## 🔄 다음 단계

### 현재 Jetson Server에서
1. Billy 서버의 Tailscale IP 확인 및 설정
2. Billy 서버에서 생성된 API Key 받기
3. 환경 변수 업데이트
4. 연결 테스트

### Billy Server에서 (해야 할 일)
1. 이 프로젝트 가져오기 (git clone 또는 rsync)
2. `.vla_aliases` 확인
3. Tailscale 설정
4. API 서버 시작
5. API Key를 Jetson에 전달

## 💡 사용 예시

### 일반적인 워크플로우

#### Jetson (데이터 수집)
```bash
vla-jetson-env       # 환경 설정
vla-network-check    # Billy 연결 확인
vla-collect          # 데이터 수집 시작
```

#### Billy (추론 제공)
```bash
vla-model-env        # 환경 설정
vla-start            # API 서버 시작
vla-status           # 상태 확인
# Jetson으로부터 추론 요청 대기 중...
```

## 🎯 주요 개선 사항

1. **명확한 역할 구분**
   - Jetson: 로봇 제어
   - Billy: 모델 추론

2. **자동 서버 감지**
   - Hostname 기반 역할 자동 설정
   - 서버별 최적화된 명령어

3. **네트워크 테스트 기능**
   - `vla-network-check`로 연결 상태 확인
   - Tailscale 통합

4. **체계적인 문서화**
   - 양쪽 서버에서 이해할 수 있는 문서
   - 단계별 설정 가이드
   - 트러블슈팅 포함

5. **동기화 용이**
   - 같은 파일을 양쪽에서 사용
   - Git 또는 rsync로 쉽게 동기화

## ⚠️ 주의사항

1. **Billy Tailscale IP 반드시 설정**
   - `.vla_aliases`에서 `100.XXX.XXX.XXX` 부분을 실제 IP로 변경

2. **API Key 안전하게 관리**
   - Jetson에서 Billy의 API Key 필요
   - 로그에서 확인 가능

3. **방화벽 설정**
   - Billy 서버에서 포트 8000 개방 필요

---

**상태**: ✅ Jetson 설정 완료 / ⏳ Billy 설정 대기  
**테스트**: ✅ Jetson에서 테스트 완료  
**버전**: 3.0 (멀티 서버)  
**작성**: 2025-12-17 19:15
