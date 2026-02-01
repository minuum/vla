#!/bin/bash
# Jetson 서버 자동 설정 스크립트
# Billy 브랜치와 통합하고 .gitignore 적용

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🤖 Jetson 서버 자동 설정"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 현재 위치 확인
if [ ! -d ".git" ]; then
    echo "❌ Git 저장소가 아닙니다"
    echo "cd ~/vla 로 이동 후 다시 실행하세요"
    exit 1
fi

# 1. 현재 상태 확인
echo "1️⃣  현재 상태 확인"
CURRENT_BRANCH=$(git branch --show-current)
echo "   현재 브랜치: $CURRENT_BRANCH"

if [ -n "$(git status --porcelain)" ]; then
    echo "   ⚠️  커밋되지 않은 변경사항이 있습니다"
else
    echo "   ✅ 깨끗한 상태"
fi
echo ""

# 2. 백업
echo "2️⃣  백업 생성"
BACKUP_DATE=$(date +%Y%m%d_%H%M%S)

# Stash 저장
if [ -n "$(git status --porcelain)" ]; then
    git stash save "auto_backup_${BACKUP_DATE}"
    echo "   ✅ 변경사항 stash에 저장"
fi

# 백업 브랜치
git branch "backup/${CURRENT_BRANCH}_${BACKUP_DATE}" 2>/dev/null || true
echo "   ✅ 백업 브랜치: backup/${CURRENT_BRANCH}_${BACKUP_DATE}"
echo ""

# 3. Billy 브랜치 가져오기
echo "3️⃣  Billy 브랜치 가져오기"
git fetch origin
echo "   ✅ Fetch 완료"

git checkout feature/inference-integration
echo "   ✅ feature/inference-integration 체크아웃"

git pull origin feature/inference-integration
echo "   ✅ 최신 코드 받기 완료"
echo ""

# 4. Git 캐시 정리 (대용량 파일)
echo "4️⃣  Git 캐시 정리 (.gitignore 적용)"
echo "   대용량 파일을 Git 추적에서 제외합니다 (파일은 유지됨)"

# 데이터셋
git rm --cached -r ROS_action/*.h5 2>/dev/null && echo "   ✅ ROS_action/*.h5 제외" || true
git rm --cached -r ROS_action/mobile_vla_dataset/*.h5 2>/dev/null && echo "   ✅ 데이터셋 제외" || true

# 체크포인트
find runs -name "*.ckpt" -type f 2>/dev/null | while read file; do
    git rm --cached "$file" 2>/dev/null || true
done
echo "   ✅ 체크포인트 제외"

# VLM 모델
git rm --cached -r .vlms/ 2>/dev/null && echo "   ✅ VLM 모델 제외" || true

# 로그
git rm --cached logs/*.log 2>/dev/null && echo "   ✅ 로그 제외" || true

# 변경사항이 있으면 커밋
if [ -n "$(git status --porcelain)" ]; then
    git commit -m "chore: Apply .gitignore for large files (Jetson)"
    echo "   ✅ .gitignore 적용 커밋"
fi
echo ""

# 5. 검증
echo "5️⃣  검증"

# Git 추적 파일 확인
TRACKED_LARGE=$(git ls-files | grep -E "(\.ckpt|\.h5)" | wc -l)
if [ "$TRACKED_LARGE" -eq 0 ]; then
    echo "   ✅ 대용량 파일이 Git에서 제외됨"
else
    echo "   ⚠️  아직 $TRACKED_LARGE 개의 대용량 파일이 Git에 추적 중"
fi

# 로컬 파일 확인
DATASET_COUNT=$(ls ROS_action/mobile_vla_dataset/*.h5 2>/dev/null | wc -l)
echo "   ✅ 로컬 데이터셋: $DATASET_COUNT 개 (유지됨)"

VLM_EXISTS=$([ -d ".vlms" ] && echo "Yes" || echo "No")
echo "   ✅ VLM 모델: $VLM_EXISTS (유지됨)"

# 클라이언트 파일 확인
if [ -f "ros2_client/vla_api_client.py" ]; then
    echo "   ✅ API 클라이언트 파일 존재"
else
    echo "   ⚠️  API 클라이언트 파일 없음"
fi
echo ""

# 6. 환경 변수 확인
echo "6️⃣  환경 변수 확인"

if [ -n "$VLA_API_SERVER" ]; then
    echo "   ✅ VLA_API_SERVER: $VLA_API_SERVER"
else
    echo "   ⚠️  VLA_API_SERVER 미설정"
    echo "   설정: export VLA_API_SERVER=\"http://100.99.189.94:8000\""
fi

if [ -n "$VLA_API_KEY" ]; then
    echo "   ✅ VLA_API_KEY: ${VLA_API_KEY:0:10}..."
else
    echo "   ⚠️  VLA_API_KEY 미설정"
    echo "   설정: export VLA_API_KEY=\"your-api-key\""
fi
echo ""

# 완료
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Jetson 설정 완료!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📝 다음 단계:"
echo ""
echo "1. 환경 변수 설정 (필요시):"
echo "   export VLA_API_SERVER=\"http://100.99.189.94:8000\""
echo "   export VLA_API_KEY=\"qwer123412341\""
echo ""
echo "2. API 클라이언트 테스트:"
echo "   python3 ros2_client/vla_api_client.py --test"
echo ""
echo "3. 이후 코드 업데이트:"
echo "   git pull origin feature/inference-integration"
echo "   # 또는"
echo "   bash scripts/sync/sync_code.sh"
echo ""
echo "📂 백업 위치:"
echo "   브랜치: backup/${CURRENT_BRANCH}_${BACKUP_DATE}"
echo "   Stash: git stash list"
echo ""
