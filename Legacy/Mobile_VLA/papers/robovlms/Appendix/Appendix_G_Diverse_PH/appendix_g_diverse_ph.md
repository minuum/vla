# RoboVLMs 논문 Appendix G: DIVERSE PH 섹션 분석

> **인용**: 논문 "APPENDIX G: DIVERSE PH" 섹션

## 1. 다양한 Policy Head 개요

### Table IX 개요
> **인용**: "TABLE IX: Sub-task level success rates by tasks in CALVIN under different training splits and VLM backbones. All models are trained with maximal 5 epochs." (논문 Appendix G 섹션)

#### Table IX의 목적
- **하위 작업 성공률**: CALVIN의 각 작업별 하위 작업 성공률
- **다양한 훈련 분할**: ABC와 ABCD 훈련 분할 비교
- **다양한 백본**: Flamingo와 KosMos 백본 비교
- **훈련 설정**: 최대 5 에포크 훈련

### 훈련 설정
- **훈련 분할**: ABC, ABCD
- **VLM 백본**: Flamingo-3B, Flamingo-4B, Flamingo-9B, KosMos
- **훈련 에포크**: 최대 5 에포크
- **평가 대상**: 34개 하위 작업

## 2. Table IX: 하위 작업별 성공률

### 모델 구성
1. **flamingo-3b-abc**: Flamingo-3B + ABC 훈련
2. **flamingo-3b-abcd**: Flamingo-3B + ABCD 훈련
3. **flamingo-4b-abcd**: Flamingo-4B + ABCD 훈련
4. **flamingo-9b-abc**: Flamingo-9B + ABC 훈련
5. **flamingo-9b-abcd**: Flamingo-9B + ABCD 훈련
6. **kosmos-abc**: KosMos + ABC 훈련
7. **kosmos-abcd**: KosMos + ABCD 훈련

### 34개 하위 작업 성능 분석

#### 작업 유형별 주요 발견사항

##### 블록 회전 작업 (6개)
- **최고 성능**: KosMos 모델들이 대부분 작업에서 95% 이상 달성
- **Flamingo 성능**: 크기별 차이 있음 (3B < 4B < 9B)
- **훈련 분할 영향**: ABCD가 ABC보다 전반적으로 높은 성능

##### 슬라이더 조작 작업 (2개)
- **move slider right**: KosMos 모델에서 100% 달성
- **move slider left**: 대부분 모델에서 95% 이상
- **Flamingo 차이**: ABC 훈련에서 성능 차이 큼

##### 블록 들기 작업 (6개)
- **KosMos 우위**: 대부분 작업에서 90% 이상 달성
- **Flamingo 성능**: 크기와 훈련 분할에 따른 차이
- **훈련 분할 효과**: ABCD 훈련이 ABC보다 우수

##### 서랍 조작 작업 (4개)
- **open/close drawer**: 대부분 모델에서 95% 이상
- **push into drawer**: 상대적으로 어려운 작업 (70-85% 범위)
- **place in drawer**: 대부분 모델에서 95% 이상

##### 조명 제어 작업 (4개)
- **turn on/off lightbulb**: KosMos 모델에서 100% 달성
- **turn on/off led**: 대부분 모델에서 95% 이상
- **Flamingo 성능**: 크기별 차이 있음

##### 블록 밀기 작업 (6개)
- **어려운 작업**: 대부분 작업에서 60-90% 범위
- **KosMos 우위**: 대부분 작업에서 80% 이상
- **방향별 차이**: left/right 방향에 따른 성능 차이

##### 복합 작업 (2개)
- **stack block**: 가장 어려운 작업 (50-90% 범위)
- **unstack block**: 상대적으로 쉬운 작업 (85-100% 범위)
- **KosMos 우위**: stack block에서 80-90% 달성

## 3. 백본별 성능 분석

### KosMos (최고 성능)
- **평균 성공률**: ~95%
- **강점**: 모든 작업 유형에서 우수한 성능
- **특징**: 대부분 작업에서 95% 이상 달성

### Flamingo 시리즈 (크기별 차이)
- **Flamingo-3B**: 기본 성능 (~80%)
- **Flamingo-4B**: 3B보다 향상 (~85%)
- **Flamingo-9B**: 가장 큰 모델 (~90%)
- **특징**: 크기가 클수록 성능 향상

### 훈련 분할의 영향
- **ABCD vs ABC**: ABCD 훈련이 전반적으로 우수
- **데이터 효과**: 더 많은 훈련 데이터의 효과
- **일반화**: ABCD 훈련이 더 나은 일반화

## 4. 훈련 분할별 성능 분석

### ABC 훈련 분할
- **특징**: 제한된 훈련 데이터
- **성능**: 상대적으로 낮은 성능
- **백본별 차이**: KosMos가 Flamingo보다 우수

### ABCD 훈련 분할
- **특징**: 더 많은 훈련 데이터
- **성능**: 전반적으로 높은 성능
- **백본별 차이**: 모든 백본에서 성능 향상

## 5. 작업별 난이도 분석

### 쉬운 작업 (높은 성공률)
- **슬라이더 조작**: 대부분 모델에서 95% 이상
- **서랍 열기/닫기**: 대부분 모델에서 95% 이상
- **조명 제어**: 대부분 모델에서 95% 이상

### 어려운 작업 (낮은 성공률)
- **블록 회전**: 일부 모델에서 80% 미만
- **서랍에 밀어넣기**: 대부분 모델에서 80% 미만
- **블록 쌓기**: 대부분 모델에서 70% 미만

### 백본별 특화 작업
- **KosMos**: 모든 작업 유형에서 우수
- **Flamingo**: 크기별 성능 차이
- **훈련 분할**: ABCD가 ABC보다 우수

## 6. 핵심 발견사항

### 백본 선택의 중요성
1. **성능 차이**: 백본에 따른 큰 성능 차이 (80-95%)
2. **작업 특화**: KosMos가 모든 작업에서 우수
3. **크기 효과**: Flamingo에서 크기별 성능 차이

### 훈련 분할의 영향
1. **데이터 효과**: ABCD 훈련이 ABC보다 우수
2. **일반화**: 더 많은 데이터가 일반화 향상
3. **백본별 차이**: 모든 백본에서 동일한 패턴

### 작업별 특성
1. **쉬운 작업**: 슬라이더, 서랍, 조명 제어
2. **어려운 작업**: 블록 회전, 쌓기, 밀기
3. **복합 작업**: stack block이 가장 어려움

## 7. 결론

### Table IX의 핵심 의의
1. **백본 선택**: VLA 성능에 결정적 영향
2. **훈련 분할**: 데이터 양이 성능에 미치는 영향
3. **성능 차이**: 백본과 훈련 분할에 따른 큰 성능 차이

### 연구의 의의
1. **체계적 비교**: 7개 모델의 34개 작업 비교
2. **실용적 가이드**: 백본과 훈련 분할 선택 가이드
3. **성능 분석**: 작업별 세부 성능 분석

### 미래 연구 방향
1. **백본 최적화**: 특정 작업에 최적화된 백본 개발
2. **성능 향상**: 낮은 성능 작업의 개선
3. **일반화**: 다양한 작업에서 안정적 성능

---

*분석 작성일: 2024년 12월*  
*원본 논문: "Towards Generalist Robot Policies: What Matters in Building Vision-Language-Action Models"*
