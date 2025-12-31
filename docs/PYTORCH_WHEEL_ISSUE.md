# PyTorch CUDA Wheel 다운로드 가이드

## 문제
NVIDIA 공식 다운로드 URL이 404 에러

## 해결 방안

### Option 1: NVIDIA Forums에서 수동 다운로드
1. 방문: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
2. JetPack 6.0 섹션에서 torch-2.3.0 wheel 찾기
3. 다운로드 후 `wheels/` 디렉토리에 저장

### Option 2: 기존 시스템 torch 활용
시스템에 이미 torch 2.7.0이 `/usr/local`에 설치되어 있음
- 문제: CPU 전용 버전
- CUDA 사용 불가

### Option 3: FP16 (FP32 없이)
BitsAndBytes INT8 대신 torch FP16 사용
- PyTorch CUDA wheel만 있으면 가능
- BitsAndBytes 불필요

## 현재 상태
- Poetry 환경: ✅ 완료
- PyTorch CUDA: ❌ 404 에러
- 다음: 수동 다운로드 필요

## 추천
1. Forums에서 wheel 수동 다운로드
2. 또는 내일 네트워크 상태 확인 후 재시도
