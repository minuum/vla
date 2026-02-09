#!/bin/bash

# 전체 실험 추론 테스트 자동화 스크립트
# 모든 학습된 모델의 PM/DA를 동일한 조건으로 측정

LOG_DIR="logs/inference_tests"
mkdir -p "$LOG_DIR"

# API Key 설정
export VLA_API_KEY="vla-mobile-fixed-key-20260205"

# 테스트할 모델 목록 (순서대로)
MODELS=(
    "exp04_baseline"
    "exp05_chunk1"
    "exp06_resampler"
    "exp09_latent128"
)

log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_DIR/test_sequence.log"
}

test_model() {
    MODEL_NAME=$1
    
    log_message "========================================="
    log_message "🧪 Testing Model: $MODEL_NAME"
    log_message "========================================="
    
    # 1. 기존 API 서버 종료
    pkill -f api_server.py
    sleep 3
    
    # 2. 해당 모델로 API 서버 시작
    log_message "🚀 Starting API server with $MODEL_NAME..."
    export VLA_MODEL_NAME="$MODEL_NAME"
    nohup python3 api_server.py > "$LOG_DIR/api_server_${MODEL_NAME}.log" 2>&1 &
    
    # 3. 모델 로딩 대기 (충분히)
    log_message "⏳ Waiting for model to load (20 seconds)..."
    sleep 20
    
    # 4. 추론 테스트 실행
    log_message "🔬 Running detailed_error_analysis.py..."
    python3 scripts/test/detailed_error_analysis.py 2>&1 | tee "$LOG_DIR/${MODEL_NAME}_accuracy_test.log"
   
    # 5. 결과 추출
    log_message "📊 Extracting results for $MODEL_NAME..."
    grep -A 30 "📊 전역 통계" "$LOG_DIR/${MODEL_NAME}_accuracy_test.log" > "$LOG_DIR/${MODEL_NAME}_summary.txt"
    
    log_message "✅ $MODEL_NAME test completed."
    log_message ""
    
    # Cooldown
    sleep 10
}

# 메인 시퀀스
log_message "🤖 All Experiments Inference Test Sequence Started"
log_message "📂 Test Dataset: ROS_action/basket_dataset_v2/test"
log_message "📊 Total Models: ${#MODELS[@]}"
log_message ""

for model in "${MODELS[@]}"; do
    test_model "$model"
done

# 최종 정리
pkill -f api_server.py

log_message "========================================="
log_message "🎉 All inference tests completed!"
log_message "========================================="
log_message "📁 Results saved in: $LOG_DIR"
log_message ""
log_message "Summary files:"
for model in "${MODELS[@]}"; do
    log_message "  - ${model}_summary.txt"
done
