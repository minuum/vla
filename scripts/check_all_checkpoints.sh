#!/bin/bash
# 모든 케이스 체크포인트 확인 스크립트

echo "========================================"
echo "체크포인트 현황 확인"
echo "========================================"
echo ""

# Case 1
echo "Case 1 (Baseline):"
find runs -name "*frozen_lora_leftright*" -name "*.ckpt" 2>/dev/null | head -3
echo ""

# Case 2
echo "Case 2 (Xavier Init):"
find runs -name "*kosmos2_fixed*" -name "*.ckpt" 2>/dev/null | head -3
echo ""

# Case 3
echo "Case 3 (Aug+Abs):"
find runs -name "*aug_abs*" -name "*.ckpt" 2>/dev/null | head -3
echo ""

# Case 4
echo "Case 4 (Right Only):"
find runs -name "*right_only*" -name "*.ckpt" 2>/dev/null | head -3
echo ""

# Case 5
echo "Case 5 (No Chunk) ⭐:"
find runs -name "*no_chunk_20251209*" -name "*.ckpt" 2>/dev/null
echo ""

echo "========================================"
echo "용량 확인"
echo "========================================"
du -sh runs/*/kosmos/mobile_vla_finetune/ 2>/dev/null | sort -h | tail -5

echo ""
echo "총 사용 공간:"
du -sh runs/ 2>/dev/null
