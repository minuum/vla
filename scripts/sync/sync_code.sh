#!/bin/bash
# Billy와 Jetson 간 코드 동기화 (Git)
# 사용법: bash scripts/sync/sync_code.sh [billy|jetson]

set -e

ROLE="${1:-auto}"  # billy, jetson, auto

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔄 Git 코드 동기화"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 역할 자동 감지
if [ "$ROLE" = "auto" ]; then
    if [ "$(hostname)" = "billy-MS-7E07" ]; then
        ROLE="billy"
    elif [ "$(hostname)" = "fevers" ]; then
        ROLE="jetson"
    else
        echo "서버 역할을 지정하세요: billy 또는 jetson"
        exit 1
    fi
fi

echo "서버 역할: $ROLE"
echo ""

# Git 상태 확인
git status --short

# 변경사항 커밋 (있으면)
if [ -n "$(git status --porcelain)" ]; then
    echo ""
    echo "⚠️  커밋되지 않은 변경사항이 있습니다"
    read -p "커밋하시겠습니까? (y/n): " COMMIT
    
    if [ "$COMMIT" = "y" ]; then
        read -p "커밋 메시지: " MESSAGE
        git add -A
        git commit -m "$MESSAGE"
    fi
fi

# Pull & Push
echo ""
echo "📥 Pulling from origin..."
git pull origin feature/inference-integration

if [ "$ROLE" = "billy" ]; then
    echo ""
    echo "📤 Pushing to origin..."
    git push origin feature/inference-integration
fi

echo ""
echo "✅ 코드 동기화 완료!"
echo ""
echo "대용량 파일 동기화:"
if [ "$ROLE" = "billy" ]; then
    echo "  체크포인트 → Jetson: bash scripts/sync/push_checkpoint_to_jetson.sh"
    echo "  데이터셋 ← Jetson: bash scripts/sync/pull_dataset_from_jetson.sh"
else
    echo "  데이터셋 → Billy: (Billy에서 pull_dataset_from_jetson.sh 실행)"
    echo "  체크포인트 ← Billy: (Billy에서 push_checkpoint_to_jetson.sh 실행)"
fi
