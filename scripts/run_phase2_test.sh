#!/bin/bash
# Phase 2: 로컬 추론 노드 테스트
# tmux 세션에서 실행

set -e

echo "========================================"
echo "  Phase 2: 로컬 추론 엔진 테스트"
echo "========================================"
echo ""

# 로그 디렉토리
LOG_DIR="logs/phase2_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "📝 로그 디렉토리: $LOG_DIR"
echo ""

# Poetry 환경 확인
echo "🔍 Poetry 환경 확인..."
poetry env info
echo ""

# PYTHONPATH 설정
echo "🔧 PYTHONPATH 설정..."
export PYTHONPATH="/home/soda/vla:/home/soda/vla/Robo+:/home/soda/vla/RoboVLMs:/home/soda/vla/RoboVLMs/robovlms:$PYTHONPATH"
echo "   ✅ PYTHONPATH 설정 완료"
echo ""

# 테스트 실행
echo "🚀 로컬 추론 엔진 테스트 시작..."
echo "⏱️  예상 시간: ~30초 (모델 로딩 포함)"
echo ""

# 테스트 실행 및 로그 저장
poetry run python test_local_inference_engine.py 2>&1 | tee "$LOG_DIR/inference_test.log"

# 결과 저장
EXIT_CODE=$?
echo ""
echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Phase 2 테스트 성공!"
    echo "✅ 로컬 추론 엔진 정상 작동"
else
    echo "❌ Phase 2 테스트 실패 (Exit Code: $EXIT_CODE)"
fi
echo "========================================"
echo ""
echo "📊 로그 파일: $LOG_DIR/inference_test.log"

exit $EXIT_CODE
