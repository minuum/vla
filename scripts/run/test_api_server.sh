#!/bin/bash
# API 서버 테스트 스크립트
# Usage: ./test_api_server.sh [health|test|predict|all]

# 색상 코드
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# API 설정
API_URL="${VLA_API_SERVER:-http://localhost:8000}"
API_KEY="${VLA_API_KEY:-jFLQzbwEch8_S2lpioP6sC-S7-Jm9MCIXpgDebrp5Uc}"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}VLA API Server Test Script${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "API URL: ${YELLOW}$API_URL${NC}"
echo -e "API Key: ${YELLOW}${API_KEY:0:20}...${NC}"
echo ""

# Health Check
test_health() {
    echo -e "${YELLOW}[1/3] Health Check${NC}"
    echo "Request: GET $API_URL/health"
    echo ""
    
    response=$(curl -s -w "\nHTTP_CODE:%{http_code}" "$API_URL/health")
    http_code=$(echo "$response" | grep HTTP_CODE | cut -d: -f2)
    body=$(echo "$response" | sed '/HTTP_CODE/d')
    
    if [ "$http_code" = "200" ]; then
        echo -e "${GREEN}✓ Status: $http_code${NC}"
        echo "$body" | python3 -m json.tool 2>/dev/null || echo "$body"
    else
        echo -e "${RED}✗ Status: $http_code${NC}"
        echo "$body"
        return 1
    fi
    echo ""
}

# Test Endpoint
test_endpoint() {
    echo -e "${YELLOW}[2/3] Test Endpoint${NC}"
    echo "Request: GET $API_URL/test"
    echo ""
    
    response=$(curl -s -w "\nHTTP_CODE:%{http_code}" \
        -H "X-API-Key: $API_KEY" \
        "$API_URL/test")
    
    http_code=$(echo "$response" | grep HTTP_CODE | cut -d: -f2)
    body=$(echo "$response" | sed '/HTTP_CODE/d')
    
    if [ "$http_code" = "200" ]; then
        echo -e "${GREEN}✓ Status: $http_code${NC}"
        echo "$body" | python3 -m json.tool 2>/dev/null || echo "$body"
    else
        echo -e "${RED}✗ Status: $http_code${NC}"
        echo "$body"
        return 1
    fi
    echo ""
}

# Predict Endpoint (with sample image)
test_predict() {
    echo -e "${YELLOW}[3/3] Predict Endpoint${NC}"
    
    # Python으로 더미 이미지 생성 및 테스트
    python3 << 'PYTHON_EOF'
import base64
import json
import requests
import sys
from io import BytesIO
from PIL import Image
import os

# API 설정
API_URL = os.getenv("VLA_API_SERVER", "http://localhost:8000")
API_KEY = os.getenv("VLA_API_KEY", "jFLQzbwEch8_S2lpioP6sC-S7-Jm9MCIXpgDebrp5Uc")

# 더미 이미지 생성 (1280x720 RGB)
print("Creating sample image (1280x720)...")
img = Image.new('RGB', (1280, 720), color=(0, 128, 255))

# Base64 인코딩
buffer = BytesIO()
img.save(buffer, format='PNG')
img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

# 요청 데이터
payload = {
    "image": img_base64,
    "instruction": "Navigate around obstacles and reach the front of the beverage bottle on the left"
}

print(f"Request: POST {API_URL}/predict")
print(f"Instruction: {payload['instruction']}")
print()

# API 호출
try:
    response = requests.post(
        f"{API_URL}/predict",
        headers={
            "X-API-Key": API_KEY,
            "Content-Type": "application/json"
        },
        json=payload,
        timeout=30
    )
    
    if response.status_code == 200:
        print(f"\033[0;32m✓ Status: {response.status_code}\033[0m")
        result = response.json()
        print(json.dumps(result, indent=2))
    else:
        print(f"\033[0;31m✗ Status: {response.status_code}\033[0m")
        print(response.text)
        sys.exit(1)
        
except requests.exceptions.Timeout:
    print("\033[0;31m✗ Request timeout (30s)\033[0m")
    sys.exit(1)
except Exception as e:
    print(f"\033[0;31m✗ Error: {e}\033[0m")
    sys.exit(1)
PYTHON_EOF
    
    if [ $? -eq 0 ]; then
        echo ""
        return 0
    else
        return 1
    fi
}

# Main
case "${1:-all}" in
    health)
        test_health
        ;;
    test)
        test_endpoint
        ;;
    predict)
        test_predict
        ;;
    all)
        test_health && test_endpoint && test_predict
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}========================================${NC}"
            echo -e "${GREEN}All tests passed! ✓${NC}"
            echo -e "${GREEN}========================================${NC}"
        else
            echo -e "${RED}========================================${NC}"
            echo -e "${RED}Some tests failed! ✗${NC}"
            echo -e "${RED}========================================${NC}"
            exit 1
        fi
        ;;
    *)
        echo "Usage: $0 [health|test|predict|all]"
        echo ""
        echo "Options:"
        echo "  health   - Health check only"
        echo "  test     - Test endpoint only"
        echo "  predict  - Predict endpoint only (requires model)"
        echo "  all      - Run all tests (default)"
        exit 1
        ;;
esac
