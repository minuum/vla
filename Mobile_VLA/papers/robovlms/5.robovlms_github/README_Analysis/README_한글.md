# RoboVLMs GitHub README 분석 (한글)

## 프로젝트 개요

**RoboVLMs**는 다양한 Vision-Language Models (VLM)을 로봇 조작 정책에 통합하여 Vision-Language-Action (VLA) 모델을 구축하기 위한 유연한 프레임워크입니다. 이 프로젝트는 최소한의 수동 설계로 사전 훈련된 VLM을 VLA로 전환하는 통합 접근 방식을 제공합니다.

## 주요 특징

### 1. 유연한 VLM 통합
- 다양한 VLM 백본 지원 (LLaVA, Flamingo, KosMos, Qwen-VL, MoonDream, PaliGemma)
- 30줄의 코드로 새로운 VLM 통합 가능
- VLM-to-VLA 변환을 위한 모듈식 아키텍처

### 2. 다양한 VLA 아키텍처
- **One-step 모델**: 단일 관찰에서 액션 예측
- **History 모델링**: 다단계 관찰 처리
- **연속/이산 액션 공간**: 유연한 액션 표현
- **정책 헤드 통합**: 효과적인 히스토리 융합 방법

### 3. 포괄적인 벤치마크
- **CALVIN**: 다중 작업 테이블탑 조작 시뮬레이션
- **SimplerEnv**: 실물-시뮬레이션 환경 평가
- **실제 실험**: 20개 작업, 74K 궤적

## 설치

### 환경 설정
```bash
# CALVIN 시뮬레이션용
conda create -n robovlms python=3.8.10 -y

# SIMPLER 시뮬레이션용
conda create -n robovlms python=3.10 -y

conda activate robovlms
conda install cudatoolkit cudatoolkit-dev -y
pip install -e .
```

### 벤치마크 환경 설정
```bash
# CALVIN 설치
bash scripts/setup_calvin.sh

# SimplerEnv 설치
bash scripts/setup_simplerenv.sh
```

### 검증
```python
# CALVIN 시뮬레이션 검증
python eval/calvin/env_test.py

# SimplerEnv 시뮬레이션 검증
python eval/simpler/env_test.py
```

## VLA 벤치마크 성능

### CALVIN 벤치마크 결과

**ABCD → D 분할:**
- KosMos P.H. (RoboVLMs): 96.7% 성공률, 4.49 평균 길이
- GR-1: 94.9% 성공률, 4.21 평균 길이
- HULC: 88.9% 성공률, 3.06 평균 길이

**ABC → D 분할:**
- KosMos P.H. (RoboVLMs): 98.0% 성공률, 4.25 평균 길이
- GR-1: 85.4% 성공률, 3.06 평균 길이
- HULC: 41.8% 성공률, 0.67 평균 길이

### SimplerEnv 성능
- WidowX+Bridge와 Google Robot 환경 모두에서 최신 성능 달성
- 다양한 로봇 플랫폼에서 강력한 일반화 능력 입증

### 실제 성능
- **20개 작업**: 포괄적인 조작 평가
- **성공률**: 75% (Simple), 60% (Unseen Distractor), 50% (Unseen Background)
- **자기 수정**: 복잡한 조작 시나리오에서 궤적 수정 기능

## VLM 통합 튜토리얼

### 1. VLM 속성 설정
VLM 통합을 위해 다음 속성들을 구성해야 합니다:

```python
# VLM 통합을 위한 핵심 속성들
@property
def image_processor(self):
    """입력 이미지 처리"""
    return self.model.processor

@property
def hidden_size(self):
    """VLM 백본의 숨겨진 크기"""
    return self.model.config.text_config.hidden_size

@property
def word_embedding(self):
    """VLM의 단어 임베딩"""
    return self.model.language_model.model.embed_tokens

@property
def text_tower(self):
    """VLM의 텍스트 처리 구성요소"""
    return self.model.language_model.model

@property
def vision_tower(self):
    """VLM의 비전 처리 구성요소"""
    return self.model.vision_tower

@property
def model(self):
    """VLM의 핵심 백본"""
    return self.backbone
```

### 2. VLA 등록
```python
# model/backbone/__init__.py에서 등록
from .robopaligemma import RoboPaligemma
__all__.append('RoboPaligemma')
```

### 3. 설정 파일 생성
```json
{
    "model": {
        "backbone": "paligemma",
        "action_head": "lstm",
        "history_length": 16,
        "action_chunk_size": 10
    },
    "training": {
        "batch_size": 128,
        "learning_rate": 1e-4,
        "epochs": 5
    }
}
```

## 훈련 파이프라인

### 데이터 전처리
- **액션 정규화**: 1st/99th 분위수로 클램핑, [-1, 1]로 정규화
- **액션 이산화**: 연속 액션을 256개 이산 토큰으로 매핑
- **히스토리 처리**: 과거 관찰의 슬라이딩 윈도우

### 모델 아키텍처
- **BaseRoboVLM**: VLM 통합을 위한 핵심 프레임워크
- **액션 헤드**: LSTM, FC, 또는 GPT 기반 액션 예측
- **멀티모달 융합**: 비전-언어-액션 통합

### 훈련 설정
```json
{
    "model": {
        "backbone": "kosmos",
        "action_head": "lstm",
        "history_length": 16,
        "action_chunk_size": 10
    },
    "training": {
        "batch_size": 128,
        "learning_rate": 1e-4,
        "epochs": 5,
        "warmup_ratio": 0.25
    }
}
```

## 평가 과정

### CALVIN 평가
- 5개 연속 작업 평가
- 1-5 작업 완료 성공률
- 평균 작업 길이 (Avg. Len.)
- ABC → D 분할 일반화

### SimplerEnv 평가
- Google Robot과 WidowX+Bridge 환경
- 다양한 작업 유형: pick, move, open/close, place
- 교차 구현 일반화 테스트

### 실제 평가
- 20개 조작 작업
- 작업당 5개 평가 설정
- 보이지 않는 객체, 배경, 기술 설명 테스트

## 지원 백본

### 인코더-디코더 아키텍처
- **Flamingo**: 교차 주의 기반 융합
- **OFA**: 통합 인코더-디코더 프레임워크

### 디코더 전용 아키텍처
- **LLaVA**: 셀프 어텐션 기반 융합
- **KosMos**: 멀티모달 트랜스포머
- **Qwen-VL**: 대규모 비전-언어 모델
- **MoonDream**: 효율적인 VLM 아키텍처
- **PaliGemma**: Google의 멀티모달 모델

## 핵심 학습 방법

### 1. 비전-언어 사전 훈련
- 대규모 웹 데이터 훈련
- 강력한 멀티모달 표현
- 로봇 조작을 위한 기초

### 2. 액션 예측
- **연속 액션**: 포즈와 그리퍼에 대한 MSE + BCE 손실
- **이산 액션**: 액션 토큰에 대한 교차 엔트로피 손실
- **히스토리 통합**: 시간 모델링을 위한 정책 헤드

### 3. 훈련 전략
- **사전 훈련**: 교차 구현 데이터 사전 훈련
- **사후 훈련**: VLM 사전 훈련 + 도메인 내 파인튜닝
- **파인튜닝**: 직접 도메인 내 훈련

## 확장성 특징

### 1. 모듈식 설계
- 쉬운 VLM 백본 교체
- 유연한 액션 헤드 선택
- 구성 가능한 훈련 파이프라인

### 2. 성능 최적화
- 메모리 효율적인 훈련
- 분산 훈련 지원
- 체크포인트 관리

### 3. 평가 프레임워크
- 다중 벤치마크 지원
- 자동화된 평가 파이프라인
- 성능 모니터링

## 문제 해결

### 일반적인 문제
1. **CUDA 호환성**: 적절한 CUDA 도구 키트 설치 확인
2. **환경 충돌**: 별도의 conda 환경 사용
3. **메모리 문제**: 배치 크기 및 모델 매개변수 조정
4. **벤치마크 설정**: 특정 설치 가이드 따르기

### 최적화 팁
1. **모델 선택**: 작업에 적합한 백본 선택
2. **하이퍼파라미터 튜닝**: 학습률과 배치 크기 최적화
3. **데이터 전처리**: 적절한 액션 정규화 보장
4. **훈련 전략**: 적절한 사전 훈련 접근법 사용

## 결론

RoboVLMs는 최소한의 수동 설계로 고성능 VLA를 구축하기 위한 포괄적인 프레임워크를 제공합니다. 프레임워크의 유연성, 포괄적인 평가, 강력한 성능은 로봇 연구 및 개발을 위한 귀중한 도구로 만듭니다.

### 주요 장점
- **쉬운 통합**: 30줄 VLM 통합 프로세스
- **강력한 성능**: 여러 벤치마크에서 최신 결과
- **유연한 아키텍처**: 다양한 VLM 백본 및 VLA 구조 지원
- **포괄적인 평가**: 다중 시뮬레이션 및 실제 벤치마크
- **오픈 소스**: 완전한 코드베이스 및 모델 가중치 사용 가능

### 응용 분야
- 일반 로봇 조작 정책
- 교차 구현 로봇 학습
- 비전-언어-액션 모델 연구
- 로봇 벤치마크 평가
- 실제 배포
