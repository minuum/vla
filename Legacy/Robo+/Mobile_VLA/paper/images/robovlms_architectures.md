# Mobile-Optimized Vision-Language-Action Model for Real-Time Robot Navigation (2-DoF Mobile Robot) - RoboVLMs 아키텍처 다이어그램

## 개요
이 다이어그램은 기존의 Generalist Policies와 최근 연구들을 RoboVLMs의 분류 체계에 따라 정리한 것입니다. RoboVLMs는 이 분류 체계를 기반으로 VLA 구조를 제시하고 실험을 진행합니다.

그림의 분류 기준은 다음과 같습니다:

* **액션 공간 (수직 축):** VLA가 예측하는 로봇 액션이 '연속적(Continuous)'인지 '이산적(Discrete)'인지에 따라 나뉩니다. 연속적인 액션은 정밀한 움직임을, 이산적인 액션은 미리 정의된 특정 동작을 의미합니다.

* **히스토리 정보 통합 방식 (수평 축):** VLA가 이전의 관찰이나 액션 이력을 어떻게 모델에 통합하는지에 따라 '원스텝(One-Step)'과 '히스토리컬(Historical)'로 나뉩니다.
  * **원스텝(One-Step):** 현재의 상태 또는 관찰만을 사용하여 다음 액션을 예측합니다.
  * **히스토리컬(Historical):** 과거의 상태나 관찰을 포함하는 슬라이딩 윈도우를 처리하여 액션을 예측합니다.
    * **Policy-Head:** 히스토리 정보가 별도의 정책 헤드를 통해 처리되고 액션 예측에 융합됩니다.
    * **Interleaved:** 히스토리컬 관찰과 액션 시퀀스가 교차된 형식으로 통합됩니다.

따라서 이 그림은 RoboVLMs가 제시하는 새로운 아키텍처라기보다는, 기존 VLA 연구들을 분석하고 분류한 결과라고 볼 수 있습니다. 이를 통해 RoboVLMs는 다양한 VLM 백본과 정책 아키텍처를 체계적으로 비교하고 평가할 수 있는 기반을 마련합니다.

![RoboVLMs 아키텍처 다이어그램](./SCR-20250828-oumx.png)

## 토큰 색상 범례
- **흰색 사각형**: Vision token
- **연한 회색 사각형**: Text token  
- **중간 회색 사각형**: History token
- **진한 회색 사각형**: Current token
- **검은색 사각형**: Discrete token

## 1. One-Step-Continuous-Action Models (상단-좌측)
현재 상태만을 사용하여 연속 액션을 한 단계에서 생성하는 모델입니다. 이는 가장 단순한 접근 방식으로, 현재 관찰만을 기반으로 다음 액션을 예측합니다.

### 입력
- VLM이 현재 상태의 토큰 시퀀스를 받습니다
- Text Tokens (연한 회색) + Vision Tokens (흰색)
- 단일 Current token (진한 회색)

### 처리 과정
1. VLM이 현재 입력 토큰을 처리
2. 단일 Current token 출력
3. Action Decoder로 전달 (점선 화살표)
4. Current action 생성

### 특징
- 히스토리 정보를 사용하지 않음
- 단순하고 빠른 처리 가능
- 복잡한 시퀀스 패턴 학습에 한계

## 2. Interleaved-Continuous-Action Models (상단-우측)
연속 액션을 교차 방식으로 생성하는 모델로, 역사적 컨텍스트를 포함합니다.

### 입력
- VLM이 교차된 토큰의 긴 시퀀스를 받습니다
- Text Tokens + Vision Tokens (반복)
- 마지막 Vision Token이 Current token

### 처리 과정
1. VLM이 교차된 입력 토큰 처리
2. History tokens + Current token 출력
3. Action Decoder로 전달
4. History token 입력과 함께 Current action 생성

## 3. One-Step-Discrete-Action Models (하단-좌측)
이산 액션을 한 단계에서 생성하는 모델입니다.

### 입력
- Text Tokens + Vision Tokens
- Discrete tokens (검은색) 시퀀스

### 처리 과정
1. VLM이 입력 토큰 처리
2. Discrete tokens 시퀀스 출력
3. Detokenizer & Reprojection 모듈로 전달
4. Discrete actions 시퀀스 생성

## 4. Policy-Head-Continuous-Action Models (하단-우측)
별도의 Policy Head를 사용하여 연속 액션을 생성하는 모델입니다.

### 입력
- Text Tokens + Vision Tokens
- 단일 Current token

### 처리 과정
1. VLM이 입력 토큰 처리
2. 단일 Current token 출력
3. Policy Head로 전달 (점선 화살표)
4. History tokens + Current action 생성

## 우리 연구와의 연관성
우리의 Mobile VLA 시스템은 **Policy-Head-Continuous-Action Models** 패러다임을 따르며, 다음과 같은 특징을 가집니다:

- **VLM**: Kosmos-2 + CLIP 하이브리드
- **Policy Head**: LSTM 기반 (4층, 4096 hidden size)
- **Action Space**: 2D 연속 액션 [linear_x, linear_y] (2-DoF 모바일 로봇)
- **실시간 처리**: 모바일 환경 최적화

이 구조는 모바일 환경에 최적화되어 실시간 로봇 제어가 가능합니다. 기존 RoboVLMs의 7-DoF 로봇 팔 제어를 2-DoF 모바일 로봇 내비게이션에 적용한 것이 우리 연구의 핵심입니다.
