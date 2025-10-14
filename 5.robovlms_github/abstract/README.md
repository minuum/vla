# RoboVLMs Abstract Analysis

## 프로젝트 개요

RoboVLMs는 Vision-Language-Action Models (VLAs)를 구축하기 위한 통합 프레임워크입니다. 이 프로젝트는 일반화된 로봇 정책을 개발하는 데 있어 핵심적인 설계 선택사항들을 체계적으로 연구합니다.

## 핵심 연구 질문

1. **왜 VLA를 선호하는가?** (Why VLAs?)
2. **어떤 백본을 선택해야 하는가?** (Which Backbone?)
3. **VLA 구조를 어떻게 공식화해야 하는가?** (How to Formulate?)
4. **언제 cross-embodiment 데이터를 활용해야 하는가?** (When to Leverage Extra Data?)

## 주요 성과

- **8개의 다양한 VLM 백본** 비교 분석
- **4가지 VLA 아키텍처** 체계적 평가
- **600개 이상의 실험** 수행
- **3개 시뮬레이션 벤치마크** 및 **실제 로봇 실험** 검증

## 실험 결과

### CALVIN 벤치마크
- **ABCD → D**: KosMos P.H. (RoboVLMs)가 96.7% 성공률 달성
- **ABC → D**: 98.0% 성공률로 기존 SOTA 대비 대폭 향상

### SimplerEnv 실험
- WidowX+Bridge 및 Google Robot 환경에서 최고 성능
- 다양한 조작 작업에서 강력한 일반화 능력 입증

### 실제 로봇 실험
- 20개 작업, 5가지 설정으로 평가
- Unseen Distractor, Unseen Background, Unseen Object, Novel Skill Description 설정에서 우수한 성능

## 핵심 발견사항

1. **VLA는 일반화된 로봇 정책을 위한 유망한 접근법**
2. **충분한 vision-language 사전 훈련이 VLA 구축에 필수적**
3. **연속 액션 공간과 policy head 구조가 최적**
4. **Cross-embodiment 데이터 활용이 few-shot 학습에 도움**

## 기술적 기여

- **RoboVLMs 프레임워크**: 30줄 이내의 코드로 VLM을 VLA로 변환
- **체계적 실험 설계**: 공정한 비교를 위한 통합 환경
- **오픈소스**: 코드, 모델, 데이터셋, 도구킷 공개

## 실용적 가치

이 연구는 VLA 설계를 위한 상세한 가이드북을 제공하며, 로봇 조작 작업에서 최고 성능을 달성하는 방법론을 제시합니다.
