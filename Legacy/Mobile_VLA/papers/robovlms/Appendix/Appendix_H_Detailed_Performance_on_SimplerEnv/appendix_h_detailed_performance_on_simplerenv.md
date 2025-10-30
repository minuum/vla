# RoboVLMs 논문 Appendix H: DETAILED PERFORMANCE ON SIMPLERENV 섹션 분석

> **인용**: 논문 "APPENDIX H: DETAILED PERFORMANCE ON SIMPLERENV" 섹션

## 1. SimplerEnv 상세 성능 개요

### Table X 및 Table XI 개요
> **인용**: "TABLE X: Simulation performance on SimplerEnv-WidowX+Bridge environments." (논문 Appendix H 섹션)

#### Table X의 목적
- **WidowX+Bridge 환경**: SimplerEnv-WidowX+Bridge 환경에서의 성능
- **하위 작업 성공률**: 각 작업의 하위 작업별 성공률
- **최종 성공률**: 전체 작업 완료 성공률
- **방법론 비교**: 다양한 로봇 방법론의 성능 비교

### Table XI 개요
> **인용**: "TABLE XI: Simulation performance on SimplerEnv-Google Robot environments." (논문 Appendix H 섹션)

#### Table XI의 목적
- **Google Robot 환경**: SimplerEnv-Google Robot 환경에서의 성능
- **다양한 작업**: 4가지 주요 작업 유형
- **방향별 성능**: 작업의 다양한 방향/조건별 성능
- **방법론 비교**: 다양한 로봇 방법론의 성능 비교

## 2. Table X: WidowX+Bridge 환경 성능

### 평가 방법론
1. **RT-1-X**: RT-1-X 방법론
2. **Octo-Base**: Octo-Base 방법론
3. **Octo-Small**: Octo-Small 방법론
4. **OpenVLA-7b**: OpenVLA-7b 방법론
5. **RoboVLMs (Ours)**: RoboVLMs 방법론

### 4개 작업 성능 분석

#### Put Spoon on Towel
- **RT-1-X**: Grasp Spoon 16.7%, final 0.0%
- **Octo-Base**: Grasp Spoon 34.7%, final 12.5%
- **Octo-Small**: Grasp Spoon 77.8%, final 47.2% (최고)
- **OpenVLA-7b**: Grasp Spoon 4.1%, final 0.0%
- **RoboVLMs**: Grasp Spoon 70.8%, final 45.8%

#### Put Carrot on Plate
- **RT-1-X**: Grasp Carrot 20.8%, final 4.2%
- **Octo-Base**: Grasp Carrot 52.8%, final 8.3%
- **Octo-Small**: Grasp Carrot 27.8%, final 9.7%
- **OpenVLA-7b**: Grasp Carrot 33.3%, final 0.0%
- **RoboVLMs**: Grasp Carrot 33.3%, final 20.8% (최고)

#### Stack Green Block on Yellow Block
- **RT-1-X**: Grasp Green Block 8.3%, final 0.0%
- **Octo-Base**: Grasp Green Block 31.9%, final 0.0%
- **Octo-Small**: Grasp Green Block 40.3%, final 4.2% (최고)
- **OpenVLA-7b**: Grasp Green Block 12.5%, final 0.0%
- **RoboVLMs**: Grasp Green Block 54.2%, final 4.2% (최고)

#### Put Eggplant in Yellow Basket
- **RT-1-X**: Grasp Eggplant 0.0%, final 0.0%
- **Octo-Base**: Grasp Eggplant 66.7%, final 43.1%
- **Octo-Small**: Grasp Eggplant 87.5%, final 56.9%
- **OpenVLA-7b**: Grasp Eggplant 8.3%, final 4.1%
- **RoboVLMs**: Grasp Eggplant 91.7%, final 79.2% (최고)

## 3. Table XI: Google Robot 환경 성능

### 평가 방법론
1. **RT-1 (Converged)**: 수렴된 RT-1
2. **RT-1 (15%)**: 15% 훈련된 RT-1
3. **RT-1-X**: RT-1-X 방법론
4. **RT-2-X**: RT-2-X 방법론
5. **Octo-Base**: Octo-Base 방법론
6. **RT-1 (Begin)**: 초기 RT-1
7. **OpenVLA-7b**: OpenVLA-7b 방법론
8. **RoboVLMs (Ours)**: RoboVLMs 방법론

### 4개 작업 성능 분석

#### Pick Coke Can
- **RT-1 (Converged)**: Horizontal 96.0%, Vertical 90.0%, Standing 71.0%, Average 85.7% (최고)
- **RT-1 (15%)**: Horizontal 86.0%, Vertical 79.0%, Standing 48.0%, Average 71.0%
- **RT-1-X**: Horizontal 82.0%, Vertical 33.0%, Standing 55.0%, Average 56.7%
- **RT-2-X**: Horizontal 74.0%, Vertical 74.0%, Standing 88.0%, Average 78.7%
- **Octo-Base**: Horizontal 21.0%, Vertical 21.0%, Standing 9.0%, Average 17.0%
- **RT-1 (Begin)**: Horizontal 5.0%, Vertical 0.0%, Standing 3.0%, Average 2.7%
- **OpenVLA-7b**: Horizontal 27.0%, Vertical 3.0%, Standing 19.0%, Average 16.3%
- **RoboVLMs**: Horizontal 94.0%, Vertical 47.0%, Standing 91.0%, Average 77.3%

#### Move Near
- **RT-1 (Converged)**: Average 44.2%
- **RT-1 (15%)**: Average 35.4%
- **RT-1-X**: Average 31.7%
- **RT-2-X**: Average 77.9% (최고)
- **Octo-Base**: Average 4.2%
- **RT-1 (Begin)**: Average 5.0%
- **OpenVLA-7b**: Average 46.2%
- **RoboVLMs**: Average 61.7%

#### Open/Close Drawer
- **RT-1 (Converged)**: Open 60.1%, Close 86.1%, Average 73.0% (최고)
- **RT-1 (15%)**: Open 46.3%, Close 66.7%, Average 56.5%
- **RT-1-X**: Open 29.6%, Close 89.1%, Average 59.7%
- **RT-2-X**: Open 15.7%, Close 34.3%, Average 25.0%
- **Octo-Base**: Open 0.9%, Close 44.4%, Average 22.7%
- **RT-1 (Begin)**: Open 0.0%, Close 27.8%, Average 13.9%
- **OpenVLA-7b**: Open 19.4%, Close 51.8%, Average 35.6%
- **RoboVLMs**: Open 33.3%, Close 53.1%, Average 43.5%

#### Open Top Drawer and Place Apple
- **RT-1 (Converged)**: Average 0.0%
- **RT-1 (15%)**: Average 0.0%
- **RT-1-X**: Average 21.3%
- **RT-2-X**: Average 3.7%
- **Octo-Base**: Average 0.0%
- **RT-1 (Begin)**: Average 0.0%
- **OpenVLA-7b**: Average 0.0%
- **RoboVLMs**: Average 24.1% (최고)

## 4. 성능 분석

### WidowX+Bridge 환경 분석

#### RoboVLMs의 성능
- **Put Spoon on Towel**: Grasp 70.8%, final 45.8% (2위)
- **Put Carrot on Plate**: Grasp 33.3%, final 20.8% (최고)
- **Stack Green Block**: Grasp 54.2%, final 4.2% (최고)
- **Put Eggplant**: Grasp 91.7%, final 79.2% (최고)

#### 경쟁 방법론 비교
- **Octo-Small**: Put Spoon on Towel에서 최고 성능
- **Octo-Base**: 전반적으로 중간 성능
- **RT-1-X**: 대부분 작업에서 낮은 성능
- **OpenVLA-7b**: 모든 작업에서 매우 낮은 성능

### Google Robot 환경 분석

#### RoboVLMs의 성능
- **Pick Coke Can**: Average 77.3% (2위)
- **Move Near**: Average 61.7% (3위)
- **Open/Close Drawer**: Average 43.5% (4위)
- **Open Top Drawer and Place Apple**: Average 24.1% (최고)

#### 경쟁 방법론 비교
- **RT-1 (Converged)**: Pick Coke Can과 Open/Close Drawer에서 최고 성능
- **RT-2-X**: Move Near에서 최고 성능
- **RT-1-X**: Open Top Drawer and Place Apple에서 2위
- **Octo-Base**: 대부분 작업에서 낮은 성능

## 5. 작업별 난이도 분석

### 쉬운 작업 (높은 성공률)
- **Pick Coke Can**: 대부분 방법론에서 50% 이상
- **Open/Close Drawer**: 일부 방법론에서 70% 이상
- **Put Eggplant**: RoboVLMs에서 79.2% 달성

### 어려운 작업 (낮은 성공률)
- **Stack Green Block**: 대부분 방법론에서 10% 미만
- **Open Top Drawer and Place Apple**: 대부분 방법론에서 25% 미만
- **Move Near**: 방법론별 큰 성능 차이

### 복합 작업의 특성
- **순차 작업**: Open Top Drawer and Place Apple이 가장 어려움
- **정밀 조작**: Stack Green Block이 가장 어려움
- **일반 조작**: Pick Coke Can이 상대적으로 쉬움

## 6. 방법론별 특성 분석

### RoboVLMs의 강점
1. **복합 작업**: Open Top Drawer and Place Apple에서 최고 성능
2. **정밀 조작**: Stack Green Block에서 최고 성능
3. **일관성**: 대부분 작업에서 상위 성능
4. **일반화**: 다양한 작업 유형에서 안정적 성능

### 경쟁 방법론의 특성
- **RT-1 (Converged)**: 기본 작업에서 우수하나 복합 작업에서 한계
- **RT-2-X**: 특정 작업에서 우수하나 전반적 성능 제한
- **Octo 시리즈**: 일부 작업에서 우수하나 일관성 부족
- **OpenVLA-7b**: 대부분 작업에서 낮은 성능

## 7. 환경별 성능 차이

### WidowX+Bridge vs Google Robot
- **WidowX+Bridge**: RoboVLMs가 대부분 작업에서 최고 성능
- **Google Robot**: RT-1 (Converged)가 기본 작업에서 우수
- **환경 특성**: 환경별로 다른 방법론의 우위

### 작업 유형별 성능
- **기본 조작**: RT-1 계열이 우수
- **복합 작업**: RoboVLMs가 우수
- **정밀 조작**: RoboVLMs가 우수

## 8. 결론

### Table X와 Table XI의 핵심 의의
1. **방법론 비교**: 다양한 로봇 방법론의 체계적 비교
2. **작업별 분석**: 세부 작업별 성능 분석
3. **환경별 차이**: 환경에 따른 성능 차이 분석

### 연구의 의의
1. **체계적 평가**: 2개 환경에서의 체계적 성능 평가
2. **실용적 가이드**: 방법론 선택에 대한 실용적 가이드
3. **성능 분석**: 작업별 세부 성능 분석

### 미래 연구 방향
1. **방법론 개선**: 낮은 성능 작업의 개선
2. **환경 적응**: 다양한 환경에서의 안정적 성능
3. **복합 작업**: 복잡한 순차 작업의 성능 향상

---

*분석 작성일: 2024년 12월*  
*원본 논문: "Towards Generalist Robot Policies: What Matters in Building Vision-Language-Action Models"*
