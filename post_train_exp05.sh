#!/bin/bash
# ============================================================
# V3-EXP-05: Post-Training Pipeline (학습 완료 후 자동 실행)
# Step 4: Best ckpt LoRA 머지
# Step 5: API 서버 exp05로 재기동 + PM/DM 평가
# ============================================================

set -e
LOGFILE="/home/billy/25-1kp/vla/logs/post_train_exp05.log"
exec > >(tee -a "$LOGFILE") 2>&1

echo "================================================================"
echo "🚀 V3-EXP-05 Post-Training Pipeline 시작"
echo "   $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"

MERGED_CKPT="/home/billy/25-1kp/vla/v3-exp05-lora/merged_v3_exp05_best.ckpt"
CONFIG_INF="/home/billy/25-1kp/vla/Mobile_VLA/configs/mobile_vla_v3_exp05_inference.json"
API_KEY="vla-mobile-fixed-key-20260205"

# ── Step 4: LoRA 머지 ──────────────────────────────────────
echo ""
echo "[ STEP 4 ] LoRA 머지 시작..."
cd /home/billy/25-1kp/vla/RoboVLMs_upstream
python3 scripts/merge_lora_exp05.py --out_path "$MERGED_CKPT"
echo "✅ Step 4 완료: $MERGED_CKPT"

# ── Step 4.5: inference config 생성 ────────────────────────
echo ""
echo "[ STEP 4.5 ] Inference config 생성..."
cd /home/billy/25-1kp/vla
python3 -c "
import json, shutil

# exp05 lora config를 기반으로 inference config 생성
with open('Mobile_VLA/configs/mobile_vla_v3_exp05_lora.json') as f:
    d = json.load(f)

# inference 전용 설정
d['exp_name'] = 'v3-exp05-lora'
d['inference_mode'] = 'classification'

# class_map 추가 (서버에서 사용)
d['class_map'] = {
    '0': [0.0, 0.0],
    '1': [1.15, 0.0],
    '2': [-1.15, 0.0],
    '3': [0.0, 1.15],
    '4': [0.0, -1.15],
    '5': [1.15, 1.15],
    '6': [1.15, -1.15],
    '7': [-1.15, 1.15],
    '8': [-1.15, -1.15]
}

with open('Mobile_VLA/configs/mobile_vla_v3_exp05_inference.json', 'w') as f:
    json.dump(d, f, indent=4, ensure_ascii=False)
print('✅ Inference config 저장: mobile_vla_v3_exp05_inference.json')
"

# ── Step 5a: API 서버 재기동 ────────────────────────────────
echo ""
echo "[ STEP 5a ] API 서버 exp05로 재기동..."

# 기존 서버 종료
pkill -f "inference_server" 2>/dev/null && sleep 3 && echo "기존 서버 종료됨" || echo "서버 없음"

export VLA_API_KEY="$API_KEY"
export VLA_CHECKPOINT_PATH="$MERGED_CKPT"
export VLA_CONFIG_PATH="$CONFIG_INF"

nohup uvicorn Mobile_VLA.inference_server:app --host 0.0.0.0 --port 8000 \
    > /home/billy/25-1kp/vla/api_server_v3_exp05.log 2>&1 &
SERVER_PID=$!
echo $SERVER_PID > /home/billy/25-1kp/vla/api_server_v3_exp05.pid
echo "✅ 서버 기동 (PID: $SERVER_PID)"
sleep 10

# health check
HEALTH=$(curl -s http://localhost:8000/health -H "X-API-Key: $API_KEY" 2>/dev/null)
echo "Health: $HEALTH"

# ── Step 5b: PM/DM 평가 ─────────────────────────────────────
echo ""
echo "[ STEP 5b ] PM/DM 비교 평가 시작..."
cd /home/billy/25-1kp/vla

export VLA_API_KEY="$API_KEY"
python3 scripts/test/api_batch_test_v3_lora.py 2>&1 | tail -30

echo ""
echo "================================================================"
echo "✅ V3-EXP-05 Pipeline 완료!"
echo "   $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"
