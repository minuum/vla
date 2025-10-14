# 🔄 One-step Models 분석

> **인용**: 논문 3페이지 2번째 줄부터 5페이지 1번째 줄까지의 Methodology 섹션

## 🎯 **One-step Models 개요**

### **One-step Modeling의 정의**
> **인용**: "one-step modeling, which utilizes only the current state or observation to produce actions" (논문에서 정의)

- **현재 상태 중심**: 현재 상태나 관측만을 사용하여 액션 생성
- **단순한 구조**: 복잡한 히스토리 처리 없이 즉시 액션 결정
- **빠른 처리**: 시간적 맥락 처리 없이 빠른 추론

## 🏗️ **One-step Models 구조**

### **기본 아키텍처**
> **인용**: 논문에서 설명된 One-step Models의 구조

- **입력**: 현재 관측 (이미지, 텍스트)
- **처리**: VLM을 통한 특징 추출 및 융합
- **출력**: 현재 시점의 액션

### **정보 흐름**
1. **Vision Input**: 현재 이미지 입력
2. **Language Input**: 현재 자연어 명령
3. **Feature Extraction**: VLM을 통한 특징 추출
4. **Action Prediction**: 즉시 액션 예측

## 🔍 **One-step Models의 특징**

### **장점**
> **인용**: 논문에서 제시된 One-step Models의 장점

- **단순성**: 복잡한 구조 없이 간단한 처리
- **속도**: 빠른 추론 속도
- **메모리 효율성**: 히스토리 저장 불필요
- **실시간 처리**: 실시간 응답 가능

### **단점**
> **인용**: 논문에서 언급된 One-step Models의 한계

- **시간적 맥락 부족**: 과거 정보 활용 불가
- **제한된 추론**: 복잡한 시퀀스 작업 어려움
- **일반화 제한**: 시간적 의존성이 있는 작업에서 성능 제한

## 📊 **One-step Models 성능**

### **적합한 작업**
> **인용**: 논문에서 제시된 One-step Models에 적합한 작업

- **즉시 반응 작업**: 현재 상태에 즉시 반응하는 작업
- **단순한 조작**: 복잡한 시퀀스가 필요 없는 작업
- **실시간 제어**: 빠른 응답이 필요한 작업

### **부적합한 작업**
> **인용**: 논문에서 언급된 One-step Models에 부적합한 작업

- **시퀀스 작업**: 시간적 순서가 중요한 작업
- **복잡한 조작**: 여러 단계가 필요한 작업
- **맥락 의존 작업**: 과거 정보가 필요한 작업

---

*분석 작성일: 2024년 12월*  
*원본 논문: "Towards Generalist Robot Policies: What Matters in Building Vision-Language-Action Models"*  
*분석자: Mobile VLA 프로젝트 팀*
