# RoboVLMs 논문 Appendix I: SUB-TASK PERFORMANCE WITH CROSS-EMBODIMENT DATASET 섹션 분석

> **인용**: 논문 "APPENDIX I: SUB-TASK PERFORMANCE WITH CROSS-EMBODIMENT DATASET" 섹션

## 1. Cross-Embodiment Dataset 성능 개요

### Table XII 및 Table XIII 개요
> **인용**: "TABLE XII: SimplerEnv simulation evaluation results for the WidowX + Bridge setup." (논문 Appendix I 섹션)

#### Table XII의 목적
- **WidowX + Bridge 설정**: SimplerEnv-WidowX + Bridge 환경에서의 성능
- **Cross-Embodiment 방법론**: 3가지 Cross-Embodiment 방법론 비교
- **하위 작업 성공률**: 각 작업의 Grasp와 final 성공률
- **방법론 비교**: Cross-Emb Pre-Train, In-domain Full Finetune, Post Train

### Table XIII 개요
> **인용**: "TABLE XIII: SimplerEnv simulator evaluation results across different policies on Google Robot tasks." (논문 Appendix I 섹션)

#### Table XIII의 목적
- **Google Robot 작업**: SimplerEnv-Google Robot 환경에서의 성능
- **4가지 방법론**: Cross-Emb Pre-Train, Target Task Finetune, In-domain Full Finetune, Post Train
- **다양한 작업**: 4가지 주요 작업 유형
- **방향별 성능**: 작업의 다양한 방향/조건별 성능

## 2. Table XII: WidowX + Bridge 설정 성능

### 평가 방법론
1. **Cross-Emb Pre-Train**: Cross-Embodiment Pre-Train 방법론
2. **In-domain Full Finetune**: In-domain Full Finetune 방법론
3. **Post Train**: Post Train 방법론

### 4개 작업 성능 분석

#### Put Spoon on Towel
- **Cross-Emb Pre-Train**: Grasp Spoon 37.5%, final 20.8%
- **In-domain Full Finetune**: Grasp Spoon 54.2%, final 29.2%
- **Post Train**: Grasp Spoon 70.8%, final 45.8% (최고)

#### Put Carrot on Plate
- **Cross-Emb Pre-Train**: Grasp Carrot 33.3%, final 25.0%
- **In-domain Full Finetune**: Grasp Carrot 25.0%, final 25.0%
- **Post Train**: Grasp Carrot 33.3%, final 20.8%

#### Stack Green Block on Yellow Block
- **Cross-Emb Pre-Train**: Grasp Green Block 8.3%, final 8.3%
- **In-domain Full Finetune**: Grasp Green Block 45.8%, final 12.5%
- **Post Train**: Grasp Green Block 54.2%, final 4.2% (최고)

#### Put Eggplant in Yellow Basket
- **Cross-Emb Pre-Train**: Grasp Eggplant 0.0%, final 0.0%
- **In-domain Full Finetune**: Grasp Eggplant 58.3%, final 58.3%
- **Post Train**: Grasp Eggplant 91.7%, final 79.2% (최고)

## 3. Table XIII: Google Robot 작업 성능

### 평가 방법론
1. **Cross-Emb Pre-Train**: Cross-Embodiment Pre-Train 방법론
2. **Target Task Finetune**: Target Task Finetune 방법론
3. **In-domain Full Finetune**: In-domain Full Finetune 방법론
4. **Post Train**: Post Train 방법론

### 4개 작업 성능 분석

#### Pick Coke Can
- **Cross-Emb Pre-Train**: Horizontal 85.0%, Vertical 43.0%, Standing 90.0%, Average 72.7%
- **Target Task Finetune**: Horizontal 21.0%, Vertical 16.0%, Standing 26.0%, Average 21.0%
- **In-domain Full Finetune**: Horizontal 92.0%, Vertical 81.0%, Standing 98.0%, Average 90.3% (최고)
- **Post Train**: Horizontal 94.0%, Vertical 47.0%, Standing 91.0%, Average 77.3%

#### Move Near
- **Cross-Emb Pre-Train**: Average 66.3%
- **Target Task Finetune**: Average 29.2%
- **In-domain Full Finetune**: Average 62.5%
- **Post Train**: Average 61.7%

#### Open/Close Drawer
- **Cross-Emb Pre-Train**: Open 28.7%, Close 25.0%, Average 26.8%
- **Target Task Finetune**: Open 16.7%, Close 66.7%, Average 41.7%
- **In-domain Full Finetune**: Open 21.3%, Close 45.4%, Average 33.3%
- **Post Train**: Open 33.3%, Close 53.1%, Average 43.5% (최고)

#### Open Top Drawer and Place Apple
- **Cross-Emb Pre-Train**: Average 36.1%
- **Target Task Finetune**: Average 0.0%
- **In-domain Full Finetune**: Average 5.6%
- **Post Train**: Average 24.1% (최고)

## 4. Cross-Embodiment 방법론별 성능 분석

### Cross-Emb Pre-Train
- **특징**: Cross-Embodiment 데이터로만 Pre-train
- **강점**: Pick Coke Can에서 중간 성능 (72.7%)
- **약점**: Put Eggplant에서 0% 성능, 대부분 작업에서 낮은 성능
- **적용성**: 기본 작업에서만 제한적 성능

### In-domain Full Finetune
- **특징**: In-domain 데이터로만 Full Finetune
- **강점**: Pick Coke Can에서 최고 성능 (90.3%)
- **약점**: 복합 작업에서 낮은 성능
- **적용성**: 특정 작업에서 우수하나 일반화 한계

### Post Train
- **특징**: Cross-Embodiment Pre-train 후 In-domain Finetune
- **강점**: 대부분 작업에서 최고 성능
- **특징**: Put Spoon on Towel (45.8%), Put Eggplant (79.2%), Open/Close Drawer (43.5%)
- **적용성**: 다양한 작업에서 안정적 성능

### Target Task Finetune
- **특징**: Target Task에만 특화된 Finetune
- **강점**: Open/Close Drawer Close에서 66.7%
- **약점**: 대부분 작업에서 매우 낮은 성능
- **적용성**: 특정 작업에만 제한적 적용

## 5. 작업별 난이도 분석

### 쉬운 작업 (높은 성공률)
- **Pick Coke Can**: 대부분 방법론에서 60% 이상
- **Move Near**: 일부 방법론에서 60% 이상
- **Put Eggplant**: Post Train에서 79.2% 달성

### 어려운 작업 (낮은 성공률)
- **Stack Green Block**: 대부분 방법론에서 15% 미만
- **Open Top Drawer and Place Apple**: 대부분 방법론에서 25% 미만
- **Put Carrot on Plate**: 대부분 방법론에서 30% 미만

### 복합 작업의 특성
- **순차 작업**: Open Top Drawer and Place Apple이 가장 어려움
- **정밀 조작**: Stack Green Block이 가장 어려움
- **일반 조작**: Pick Coke Can이 상대적으로 쉬움

## 6. Cross-Embodiment 데이터의 영향

### Pre-train의 효과
- **제한적 효과**: Cross-Emb Pre-Train만으로는 부족
- **기본 작업**: Pick Coke Can에서만 중간 성능
- **복합 작업**: 대부분 작업에서 낮은 성능

### Post-train의 효과
- **최고 성능**: 대부분 작업에서 최고 성능
- **일반화**: 다양한 작업에서 안정적 성능
- **복합 작업**: Put Eggplant에서 79.2% 달성

### In-domain 데이터의 중요성
- **특정 작업**: Pick Coke Can에서 최고 성능
- **제한적 일반화**: 다른 작업에서 성능 제한
- **방법론별 차이**: 작업별로 다른 방법론의 우위

## 7. 방법론별 특성 분석

### Post Train의 우수성
1. **복합 작업**: Put Eggplant에서 79.2% 달성
2. **일관성**: 대부분 작업에서 상위 성능
3. **일반화**: 다양한 작업 유형에서 안정적 성능
4. **Cross-Embodiment 효과**: Pre-train과 Finetune의 시너지

### In-domain Full Finetune의 특성
- **특정 작업**: Pick Coke Can에서 최고 성능 (90.3%)
- **제한적 일반화**: 다른 작업에서 성능 제한
- **방법론별 차이**: 작업별로 다른 방법론의 우위

### Cross-Emb Pre-Train의 한계
- **제한적 성능**: 대부분 작업에서 낮은 성능
- **기본 작업**: Pick Coke Can에서만 중간 성능
- **복합 작업**: Put Eggplant에서 0% 성능

## 8. 환경별 성능 차이

### WidowX + Bridge vs Google Robot
- **WidowX + Bridge**: Post Train이 대부분 작업에서 최고 성능
- **Google Robot**: In-domain Full Finetune이 Pick Coke Can에서 최고 성능
- **환경 특성**: 환경별로 다른 방법론의 우위

### 작업 유형별 성능
- **기본 조작**: In-domain Full Finetune이 우수
- **복합 작업**: Post Train이 우수
- **정밀 조작**: Post Train이 우수

## 9. 결론

### Table XII와 Table XIII의 핵심 의의
1. **Cross-Embodiment 효과**: Cross-Embodiment 데이터의 효과 분석
2. **방법론 비교**: 4가지 Cross-Embodiment 방법론의 체계적 비교
3. **작업별 분석**: 세부 작업별 성능 분석

### 연구의 의의
1. **체계적 평가**: 2개 환경에서의 체계적 성능 평가
2. **실용적 가이드**: Cross-Embodiment 방법론 선택에 대한 실용적 가이드
3. **성능 분석**: 작업별 세부 성능 분석

### 핵심 발견사항
1. **Post Train 우수성**: 대부분 작업에서 최고 성능
2. **Cross-Embodiment 효과**: Pre-train과 Finetune의 시너지
3. **작업별 차이**: 작업 유형별로 다른 방법론의 우위

### 미래 연구 방향
1. **방법론 개선**: 낮은 성능 작업의 개선
2. **Cross-Embodiment 최적화**: 더 효과적인 Cross-Embodiment 방법론 개발
3. **복합 작업**: 복잡한 순차 작업의 성능 향상

---

*분석 작성일: 2024년 12월*  
*원본 논문: "Towards Generalist Robot Policies: What Matters in Building Vision-Language-Action Models"*
