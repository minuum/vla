# Simple CLIP LSTM 모델 체크포인트 기반 추론 시스템 개발 진행상황

## 📋 프로젝트 개요

**목표**: `best_simple_clip_lstm_model.pth` 체크포인트를 기반으로 도커 환경에서 추론 시스템 구축

**체크포인트 위치**: `vla/Robo+/Mobile_VLA/simple_clip_lstm_model/best_simple_clip_lstm_model.pth` (7.3GB)

## 🔍 모델 구조 분석 결과

### 실제 모델 구조 (체크포인트 키 분석)
- **Kosmos2 + CLIP 하이브리드 모델**
- **Kosmos2 Text Model**: 24층 Transformer (4096차원)
- **Kosmos2 Vision Model**: 24층 ViT (4096차원)
- **CLIP Text Model**: 12층 Transformer (768차원)
- **CLIP Vision Model**: 12층 ViT (768차원)
- **Feature Fusion Layer**: 4864 → 2048
- **LSTM Layer**: 4층 (512차원)
- **Action Head**: 2차원 출력 (로봇 제어)

### 체크포인트 키 구조
```
kosmos_model.text_model.model.layers.0-23.*
kosmos_model.vision_model.model.encoder.layers.0-23.*
clip_model.text_model.encoder.layers.0-11.*
clip_model.vision_model.encoder.layers.0-11.*
feature_fusion.weight/bias
rnn.weight_ih_l0-3, rnn.weight_hh_l0-3
actions.mlp.0-8.weight/bias
```

## 🚀 개발된 시스템 구성

### 1. 메인 추론 스크립트
- **`vla/run_simple_clip_lstm_inference.sh`**: 도커 컨테이너 실행 및 추론 시작
- **`vla/kosmos_clip_hybrid_inference.py`**: Kosmos2 + CLIP 하이브리드 모델 추론
- **`vla/memory_optimized_inference.py`**: 메모리 최적화된 추론 (메모리 모니터링 포함)

### 2. 도커 환경 설정
- **베이스 이미지**: `mobile_vla:pytorch-2.3.0-cuda`
- **GPU 지원**: Jetson Orin NX (CUDA 지원)
- **볼륨 마운트**: 체크포인트 파일 접근

## ⚠️ 발생한 문제점들

### 1. 메모리 부족 문제
- **증상**: 모델 로드 시 "Killed" 오류
- **원인**: 7.3GB 체크포인트 + 대규모 모델 구조
- **해결책**: 메모리 최적화된 모델 구조 구현

### 2. 모델 구조 불일치
- **증상**: `RuntimeError: Missing key(s) in state_dict`
- **원인**: 예상 모델 구조와 실제 체크포인트 구조 불일치
- **해결책**: 실제 체크포인트 키 구조에 맞는 모델 정의

### 3. 컨테이너 실행 문제
- **증상**: 컨테이너 종료 및 재시작 필요
- **원인**: 메모리 부족으로 인한 프로세스 종료
- **해결책**: 메모리 모니터링 및 최적화

## 🔧 구현된 해결책

### 1. 메모리 최적화
```python
# 메모리 모니터링
def get_memory_info():
    # 시스템 메모리 + GPU 메모리 정보

# 메모리 정리
def clear_memory():
    torch.cuda.empty_cache()
    gc.collect()

# 모델 최적화
- 레이어 수 줄임 (24→6, 12→6)
- 입력 크기 최적화
- 중간 텐서 메모리 해제
```

### 2. 모델 구조 수정
```python
class MemoryOptimizedKosmosCLIPModel(nn.Module):
    # Kosmos2 모델 (6층으로 축소)
    # CLIP 모델 (6층으로 축소)
    # LSTM (2층으로 축소)
    # 메모리 효율적인 forward pass
```

### 3. 대화형 추론 시스템
```python
# 사용 가능한 명령어
- 'infer': 단일 추론 실행
- 'benchmark': 성능 벤치마크
- 'memory': 메모리 상태 확인
- 'clear': 메모리 정리
- 'quit': 종료
```

## 📊 현재 상태

### ✅ 완료된 작업
1. 체크포인트 구조 분석
2. 메모리 최적화된 모델 구현
3. 도커 컨테이너 실행 스크립트
4. 메모리 모니터링 시스템
5. 대화형 추론 인터페이스

### ⚠️ 현재 문제점
1. 모델 가중치 로드 시 구조 불일치
2. 메모리 사용량이 여전히 높음
3. 실제 추론 성능 테스트 미완료

### 🔄 다음 단계
1. 실제 체크포인트 구조에 맞는 모델 수정
2. 메모리 사용량 추가 최적화
3. 실제 추론 성능 테스트
4. 로봇 제어 연동

## 🎯 성능 목표

### 예상 성능
- **모델 크기**: 대규모 하이브리드 (Kosmos2 + CLIP)
- **입력**: Vision patches + Text tokens
- **출력**: 2D 로봇 액션 (선형/각속도)
- **목표 FPS**: Jetson Orin NX에서 실시간 추론

### 메모리 요구사항
- **체크포인트**: 7.3GB
- **모델 로드**: ~8-10GB
- **추론 시**: ~2-4GB
- **총 필요**: ~12-15GB

## 📝 실행 방법

### 1. 추론 컨테이너 시작
```bash
cd /home/soda
./vla/run_simple_clip_lstm_inference.sh
```

### 2. 메모리 최적화된 추론 실행
```bash
python3 vla/memory_optimized_inference.py
```

### 3. 메모리 상태 확인
```bash
# 컨테이너 내부에서
python3 -c "import psutil; print(psutil.virtual_memory())"
nvidia-smi
```

## 🔍 디버깅 정보

### 체크포인트 분석
- **파일 크기**: 7.3GB
- **키 수**: 1000+ 개
- **주요 구성**: Kosmos2 + CLIP + LSTM + Action Head

### 메모리 사용량
- **시스템 메모리**: 16GB (Jetson Orin NX)
- **GPU 메모리**: 8GB
- **현재 사용량**: 모델 로드 시 90%+ 사용

### 오류 로그
```
RuntimeError: Error(s) in loading state_dict for SimpleCLIPLSTMModel:
Missing key(s) in state_dict: "vision_encoder.weight", ...
Unexpected key(s) in state_dict: "kosmos_model.text_model.model.embed_tokens.weight", ...
```

## 📈 향후 계획

### 단기 목표 (1-2주)
1. 모델 구조 완전 수정
2. 메모리 최적화 완료
3. 기본 추론 성능 확인

### 중기 목표 (1개월)
1. 실시간 추론 성능 달성
2. 로봇 제어 연동
3. 실제 환경 테스트

### 장기 목표 (3개월)
1. 성능 최적화 완료
2. 배포 시스템 구축
3. 문서화 및 가이드 완성

---

**마지막 업데이트**: 2024-08-24
**상태**: 개발 중 (메모리 최적화 단계)
**다음 마일스톤**: 모델 구조 수정 및 추론 성공
