#!/bin/bash

# =============================================================================
# 🚀 Mobile VLA Docker 빌드 스크립트 - 검증된 VLA 환경 기반
# =============================================================================

set -e

# 색상 코드
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}🔨 Mobile VLA Docker 이미지 빌드 시작 (검증된 VLA 환경 기반)${NC}"
echo -e "${YELLOW}⚠️  베이스 이미지 크기: ~43.7GB (nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3)${NC}"
echo -e "${BLUE}📦 이미지명: mobile_vla:verified-base${NC}"
echo

# pytorch_cuda_test.py 파일 존재 확인
if [ ! -f "pytorch_cuda_test.py" ]; then
    echo -e "${YELLOW}⚠️  pytorch_cuda_test.py 파일이 없습니다. 생성 중...${NC}"
    cat > pytorch_cuda_test.py << 'EOF'
#!/usr/bin/env python3
import torch
import sys

def main():
    print("🔍 PyTorch & CUDA 테스트")
    print(f"PyTorch 버전: {torch.__version__}")
    print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA 버전: {torch.version.cuda}")
        print(f"CUDA 디바이스 수: {torch.cuda.device_count()}")
        print(f"현재 CUDA 디바이스: {torch.cuda.current_device()}")
        print(f"디바이스 이름: {torch.cuda.get_device_name(0)}")
        
        # 간단한 CUDA 연산 테스트
        try:
            x = torch.rand(5, 3).cuda()
            y = torch.rand(5, 3).cuda()
            z = x + y
            print("✅ CUDA 연산 테스트 성공")
        except Exception as e:
            print(f"❌ CUDA 연산 테스트 실패: {e}")
    else:
        print("❌ CUDA를 사용할 수 없습니다")
    
    print("🎉 테스트 완료")

if __name__ == "__main__":
    main()
EOF
    chmod +x pytorch_cuda_test.py
    echo -e "${GREEN}✅ pytorch_cuda_test.py 생성 완료${NC}"
fi

# Docker 빌드 시작
echo -e "${BLUE}🔨 Docker 이미지 빌드 중... (시간이 오래 걸릴 수 있습니다)${NC}"

docker-compose -f docker-compose.mobile-vla.yml build --no-cache mobile-vla

if [ $? -eq 0 ]; then
    echo
    echo -e "${GREEN}✅ Mobile VLA Docker 이미지 빌드 완료!${NC}"
    echo -e "${BLUE}📋 다음 단계:${NC}"
    echo "   1️⃣  컨테이너 실행: ./docker-run-verified.sh"
    echo "   2️⃣  컨테이너 접속: docker exec -it mobile_vla_verified bash"
    echo "   3️⃣  CUDA 테스트: docker exec -it mobile_vla_verified torch_cuda_test"
    echo "   4️⃣  카메라 테스트: docker exec -it mobile_vla_verified vla-camera"
    echo
    echo -e "${BLUE}🔍 이미지 정보:${NC}"
    docker images | grep mobile_vla
else
    echo -e "${RED}❌ Docker 이미지 빌드 실패!${NC}"
    exit 1
fi