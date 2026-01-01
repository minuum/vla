# Phase 2 테스트 실행 가이드

**일시**: 2026-01-02 03:33 KST  
**상태**: ✅ 세션 안전하게 실행 중

---

## ✅ 세션 관리

### 현재 실행 중인 세션
```bash
tmux ls
# 출력: phase2_test: 1 windows (created Fri Jan  2 03:33:21 2026)
```

### 세션에 접속하는 방법
```bash
# 실시간으로 테스트 진행 상황 보기
tmux attach -t phase2_test

# 세션에서 나오기 (종료하지 않고)
# Ctrl+B, D (detach)
```

### 세션 종료 방지
이번에는 **`exec bash`**를 추가하여:
- 테스트 완료 후에도 세션 유지 ✅
- Antigravity IDE 끊겨도 세션 계속 실행 ✅
- 언제든지 다시 접속 가능 ✅

---

## 🔄 현재 진행 상황

### 1️⃣ 세션 시작 (03:33:21)
```bash
✅ tmux 세션 생성 및 시작
✅ 로그 파일 자동 저장: logs/phase2_test_20260102_033320.log
```

### 2️⃣ 진행 단계
| 단계 | 상태 | 예상 시간 |
|------|------|-----------|
| Tokenizer 로딩 | ✅ 완료 | ~5초 |
| MobileVLATrainer import | ⏳ 진행중 | 1-2분 |
| Kosmos-2 로딩 | 대기중 | 1-2분 |
| 체크포인트 로드 | 대기중 | 1-2분 |
| FP16 변환 + GPU 전송 | 대기중 | 1분 |
| 추론 테스트 | 대기중 | 즉시 |

**총 예상 시간**: 5-8분

---

## 📋 모니터링 명령어

### 실시간 확인
```bash
# 1. tmux 세션 직접 보기 (추천)
tmux attach -t phase2_test

# 2. 로그 파일 실시간 확인
tail -f logs/phase2_test_20260102_033320.log

# 3. tmux 화면 캡처
tmux capture-pane -t phase2_test -p | tail -30

# 4. 세션 상태 확인
tmux ls
```

### 시스템 상태 확인
```bash
# CPU & 메모리
htop

# GPU (Jetson)
nvidia-smi  # 또는 jtop

# 프로세스 확인
ps aux | grep python
```

---

## 🎯 예상 출력 순서

```
🚀 Phase 2 테스트 시작...
Special tokens have been added...          ← [완료]
======================================================================  
  로컬 추론 엔진 테스트
======================================================================

📦 [1/5] Config 생성...
   ✅ Window: 2, Chunk: 10

🚀 [2/5] 추론 엔진 초기화...
🔧 Device: cuda
🎮 GPU: Orin
   ✅ 엔진 생성 완료

📦 [3/5] 모델 로딩 중...                 ← [현재 예상 위치]
🚀 모델 로딩: runs/.../chunk5/epoch_...
⚙️  메모리 최적화 모드: CPU 로드 → FP16 → GPU
📥 체크포인트 로딩 (CPU)...
📝 모델 설정: model_name=...
🔧 MobileVLATrainer 초기화...
📦 State dict 로딩...
🔄 FP16 변환 중...
🎮 GPU로 전송 중...
✅ 모델 로드 완료!
💾 GPU 메모리: X.XX GB

🖼️  [4/5] 테스트 이미지 준비...
   ✅ 이미지 버퍼 준비 완료

🎯 [5/5] 추론 실행...
   ✅ 추론 성공!
   시간: XXX ms
   액션 예측: [X.XXX, X.XXX]

======================================================================
✅ 로컬 추론 엔진 테스트 성공!
======================================================================

✅ 테스트 완료! 세션을 유지합니다.
tmux attach -t phase2_test 로 다시 접속 가능
```

---

## ⚠️ 만약 문제가 생긴다면

### 세션이 응답 없을 때
```bash
# 1. 프로세스 확인
ps aux | grep python

# 2. 강제 종료 후 재시작
tmux kill-session -t phase2_test
# 다시 실행
```

### 로그에서 에러 확인
```bash
# 로그 파일 전체 보기
cat logs/phase2_test_20260102_033320.log

# 에러만 필터링
cat logs/phase2_test_20260102_033320.log | grep -i "error\|fail\|exception"
```

---

## 🚀 세션 유지 기능

이번에 추가한 기능:
```bash
exec bash  # 테스트 완료 후 bash 셸 유지
```

**장점**:
- ✅ 테스트 완료 후에도 세션 자동 종료 안 됨
- ✅ 결과를 직접 확인하고 추가 명령 실행 가능
- ✅ Antigravity IDE 재연결 후에도 세션 유지
- ✅ 네트워크 끊김에도 작업 계속 진행

---

## 📞 도움말

### 세션 관련
- **접속**: `tmux attach -t phase2_test`
- **나가기**: `Ctrl+B, D`
- **종료**: `tmux kill-session -t phase2_test`
- **목록**: `tmux ls`

### 로그 관련
- **실시간**: `tail -f logs/phase2_test_20260102_033320.log`
- **전체**: `cat logs/phase2_test_20260102_033320.log`
- **최근**: `tail -50 logs/phase2_test_20260102_033320.log`

---

**현재**: 정상 진행 중 ✅  
**다음**: 5-8분 후 완료 예상
