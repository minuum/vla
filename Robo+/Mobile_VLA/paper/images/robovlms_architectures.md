# Mobile-Optimized Vision-Language-Action Model for Real-Time Robot Navigation - RoboVLMs 아키텍처 다이어그램

## 개요
이 다이어그램은 RoboVLMs의 4가지 주요 아키텍처 패러다임을 보여줍니다. 각 모델은 액션 생성 방식에 따라 분류됩니다.

![RoboVLMs 아키텍처 다이어그램](./SCR-20250828-oumx.png)

## 토큰 색상 범례
- **흰색 사각형**: Vision token
- **연한 회색 사각형**: Text token  
- **중간 회색 사각형**: History token
- **진한 회색 사각형**: Current token
- **검은색 사각형**: Discrete token

## 1. One-Step-Continuous-Action Models (상단-좌측)
단일 연속 액션을 한 단계에서 생성하는 모델입니다.

### 입력
- VLM이 토큰 시퀀스를 받습니다
- Text Tokens (연한 회색) + Vision Tokens (흰색)
- 단일 Current token (진한 회색)

### 처리 과정
1. VLM이 입력 토큰을 처리
2. 단일 Current token 출력
3. Action Decoder로 전달 (점선 화살표)
4. Current action 생성

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
- **Action Space**: 2D 연속 액션 [linear_x, linear_y]
- **실시간 처리**: 750+ FPS 달성

이 구조는 모바일 환경에 최적화되어 실시간 로봇 제어가 가능합니다.
