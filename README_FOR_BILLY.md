# 🚨 Billy 서버 담당자님께

## 📢 중요 업데이트: VLA 멀티 서버 환경 구성

Jetson 서버에서 VLA 시스템을 **멀티 서버 구조**로 재구성했습니다!
Billy 서버에서도 설정이 필요합니다.

---

## 🔥 Billy 서버에서 해야 할 일

### 1️⃣ 최신 코드 받기
```bash
cd ~/vla  # 또는 프로젝트 디렉토리
git checkout feature/inference-integration
git pull
```

### 2️⃣ 설정 가이드 확인
다음 파일을 순서대로 읽어주세요:

1. **`BILLY_SETUP.md`** ← ⭐ 여기서 시작!
   - Billy 서버 전용 단계별 설정 가이드

2. **`docs/VLA_MULTI_SERVER_SETUP.md`**
   - 전체 시스템 아키텍처 및 상세 문서

3. **`QUICK_REFERENCE.md`**
   - 빠른 참조 카드

---

## 📋 간단 요약

### 새로운 시스템 구조
```
Jetson (fevers)          ←→          Billy (당신의 서버)
로봇 서버                Tailscale     모델 서버
100.85.118.58                         ???.???.???.???

• 이미지 캡쳐                          • VLA 모델 추론
• 로봇 주행                            • API 서버
• 데이터 수집                          • GPU 연산
```

### 필요한 설정

1. **Tailscale 설치 및 IP 확인**
   ```bash
   tailscale ip -4
   # 예: 100.123.45.67
   ```

2. **환경 변수 설정**
   ```bash
   # ~/.zshrc 또는 ~/.bashrc에 추가
   export BILLY_TAILSCALE_IP="100.123.45.67"  # 위에서 확인한 IP
   ```

3. **소스 aliases 파일**
   ```bash
   echo 'source ~/vla/.vla_aliases' >> ~/.zshrc
   source ~/.zshrc
   ```

4. **API 서버 시작**
   ```bash
   vla-model-env    # 환경 설정
   vla-start        # API 서버 시작
   vla-status       # 상태 확인
   ```

5. **API Key 확인 및 Jetson에 전달**
   ```bash
   vla-logs | grep "API Key"
   # 출력된 API Key를 Jetson 담당자에게 알려주세요
   ```

---

## 🎯 설정 완료 후

### 사용 가능한 명령어
```bash
vla-model-env        # 모델 환경 설정
vla-start            # API 서버 시작
vla-stop             # API 서버 중지
vla-restart          # API 서버 재시작
vla-status           # API 서버 상태
vla-logs             # API 서버 로그
vla-test             # API 전체 테스트
vla-health           # Health check
vla-network-check    # Jetson 연결 확인
vla-env              # 환경 변수 확인
vla-help             # 도움말
```

---

## 📞 연락처

- **Jetson 서버 Tailscale IP**: `100.85.118.58`
- **필요한 정보**:
  - Billy Tailscale IP
  - API Key (API 서버 시작 후 생성됨)

---

## 📄 커밋 정보

- **Commit**: `90ca4532`
- **Branch**: `feature/inference-integration`
- **Date**: 2025-12-17 19:21
- **Message**: "feat: VLA 멀티 서버 환경 구조 재구성 (Jetson-Billy 분리)"

---

**설정에 문제가 있으시면 `BILLY_SETUP.md`를 참조하거나 연락주세요!** 🙏
