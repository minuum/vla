# RoboVLMs Introduction Analysis

## 연구 배경

### Vision-Language-Action Models (VLAs)의 중요성
- 인간의 지시에 따라 물리적 환경을 인식, 추론, 상호작용할 수 있는 일반화된 로봇 정책 구축
- Vision-Language Models (VLMs)를 로봇 데이터로 파인튜닝하여 VLA로 변환하는 접근법이 주목받음

### 기존 연구의 한계
1. **일관성 없는 VLA 정의**: 다양한 연구에서 VLA의 엄격한 정의가 일치하지 않음
2. **체계적 이해 부족**: VLA 설계 선택사항에 대한 체계적 이해 부족
3. **비교 연구 부족**: 다양한 백본, 구조, 데이터 분포, 훈련 방법론의 공정한 비교 부족

## 연구 목표

### 핵심 연구 질문
1. **왜 VLA를 선호하는가?**
   - 다른 일반화 정책 대비 VLA의 장점
   - 실제 시나리오에서의 VLA 성능

2. **어떤 백본을 선택해야 하는가?**
   - 다양한 VLM 백본의 VLA 구축 적합성
   - Vision-language 사전 훈련의 영향

3. **VLA 구조를 어떻게 공식화해야 하는가?**
   - 최적의 VLA 아키텍처
   - 일반화 및 데이터 효율성에 미치는 영향

4. **언제 cross-embodiment 데이터를 활용해야 하는가?**
   - 대규모 cross-embodiment 데이터셋의 기여도
   - 데이터 활용 전략

## VLA 구조 분류

### 1. 히스토리 정보 모델링
- **One-step modeling**: 현재 상태만 활용
- **History modeling**: 슬라이딩 윈도우의 히스토리 상태 활용

### 2. 히스토리 정보 집계 방법
- **Interleaved modeling**: 관찰과 액션 시퀀스를 교차 형식으로 통합
- **Policy head**: 각 히스토리 단계를 별도 처리하고 정책 헤드에서 정보 융합

### 3. 액션 공간
- **Continuous**: 연속 액션 공간
- **Discrete**: 이산 액션 공간

## 실험 설계

### 벤치마크 선택
1. **CALVIN**: 시뮬레이션 멀티태스크 테이블탑 조작
2. **SimplerEnv**: 실제-시뮬 환경
3. **실제 로봇 실험**: 100개 조작 작업, 74K 궤적

### 평가 설정
- **ABCD/ABC 분할**: 일반화 능력 평가
- **다양한 설정**: Unseen Distractor, Unseen Background, Unseen Object, Novel Skill Description

## 기대 성과

이 연구를 통해 VLA 구축의 핵심 요소들을 체계적으로 이해하고, 최적의 VLA 설계 가이드라인을 제공할 것으로 기대됩니다.
