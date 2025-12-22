# Jetson 서버 브랜치 통합 가이드

**현재 상황**: Jetson은 `feature/mobile-vla-refactor`, Billy는 `feature/inference-integration`  
**목표**: 안전하게 브랜치 통합 및 .gitignore 적용

---

## 🚀 빠른 실행 (자동)

```bash
cd ~/vla
bash scripts/jetson_sync_setup.sh
```

---

## 📋 수동 실행 (단계별)

### 1단계: 현재 상태 백업

```bash
cd ~/vla

# 현재 브랜치 확인
git branch

# 작업 중인 변경사항 저장
git stash save "backup_before_merge_$(date +%Y%m%d_%H%M%S)"

# 백업 브랜치 생성
git branch backup/mobile-vla-refactor-$(date +%Y%m%d)
```

### 2단계: Billy 브랜치 가져오기

```bash
# 최신 상태 가져오기
git fetch origin

# Billy 브랜치로 전환
git checkout feature/inference-integration

# 최신 코드 받기
git pull origin feature/inference-integration
```

### 3단계: .gitignore 적용 (대용량 파일 제외)

```bash
# Git 캐시에서 대용량 파일 제거 (파일은 유지됨)
git rm --cached -r ROS_action/*.h5 2>/dev/null || true
git rm --cached -r ROS_action/mobile_vla_dataset/*.h5 2>/dev/null || true
git rm --cached -r runs/**/*.ckpt 2>/dev/null || true
git rm --cached -r .vlms/ 2>/dev/null || true
git rm --cached logs/*.log 2>/dev/null || true

# 변경사항 확인
git status
```

### 4단계: 기존 브랜치 내용 병합 (필요시)

```bash
# mobile-vla-refactor에만 있던 중요한 내용이 있다면
git checkout backup/mobile-vla-refactor-$(date +%Y%m%d)

# 필요한 파일 확인
git log --oneline -10

# 필요한 파일만 선택적으로 가져오기
git checkout feature/inference-integration
git checkout backup/mobile-vla-refactor-20251216 -- path/to/important/file
```

### 5단계: 정리 및 확인

```bash
# 현재 브랜치 확인
git branch

# Git이 추적하지 않는 파일 확인 (정상)
git status --ignored

# 실제 파일은 여전히 존재 (로컬에만)
ls -la ROS_action/mobile_vla_dataset/ | head
ls -la .vlms/
```

---

## ✅ 검증

### Git 상태 확인
```bash
# 추적 중인 파일 확인
git ls-files | grep -E "(\.ckpt|\.h5|\.log)"
# 결과: 없어야 정상

# .gitignore 작동 확인
git check-ignore -v ROS_action/test.h5
# 결과: .gitignore:X:... 출력되면 정상
```

### 파일 존재 확인
```bash
# 데이터셋은 로컬에 그대로 있음
ls -lh ~/vla/ROS_action/mobile_vla_dataset/*.h5 | wc -l

# 코드는 최신 버전
ls -la ~/vla/ros2_client/vla_api_client.py
```

---

## 🔄 이후 작업 흐름

### 코드 업데이트 (Git)
```bash
cd ~/vla
git pull origin feature/inference-integration

# 또는
bash scripts/sync/sync_code.sh
```

### 데이터셋 관리 (로컬)
```bash
# Billy로 전송할 때만 (Billy에서 실행)
# bash scripts/sync/pull_dataset_from_jetson.sh
```

### 체크포인트 받기 (Billy에서 전송)
```bash
# Billy에서:
# bash scripts/sync/push_checkpoint_to_jetson.sh

# Jetson에서 확인:
ls -lh ~/vla/ROS_action/last.ckpt
```

---

## ⚠️ 주의사항

1. **데이터 손실 없음**
   - `.gitignore`는 Git 추적만 중지
   - 실제 파일은 로컬에 그대로 유지

2. **백업 브랜치**
   - `backup/mobile-vla-refactor-*` 브랜치는 삭제하지 말 것
   - 혹시 모를 상황 대비

3. **Push하지 말 것**
   - Jetson에서는 `git pull`만 사용
   - `git push`는 Billy에서만

4. **대용량 파일**
   - Git으로 관리 ❌
   - rsync로 관리 ✅

---

## 🆘 문제 해결

### 충돌 발생 시
```bash
# 변경사항 모두 버리고 Billy 버전 사용
git reset --hard origin/feature/inference-integration
```

### 실수로 대용량 파일 add한 경우
```bash
# 스테이징에서 제거
git reset HEAD ROS_action/*.h5

# 또는 전체 초기화
git reset HEAD .
```

### 브랜치 돌아가기
```bash
# 백업 브랜치로 복구
git checkout backup/mobile-vla-refactor-20251216

# 다시 시도
git checkout feature/inference-integration
```

---

## 📝 체크리스트

### 실행 전
- [ ] 중요한 변경사항 백업 확인
- [ ] 현재 브랜치 확인

### 실행 중
- [ ] git stash로 작업 저장
- [ ] 백업 브랜치 생성
- [ ] feature/inference-integration 체크아웃
- [ ] .gitignore 적용
- [ ] Git 캐시 정리

### 실행 후
- [ ] git status로 대용량 파일 제외 확인
- [ ] 로컬 파일 존재 확인
- [ ] 클라이언트 파일 테스트

---

**작성**: 2025-12-16 23:46  
**Jetson 서버 전용**
