# 📊 논문 그림/Figure 제안사항

## 🎯 **최소한의 필수 그림들**

### **Figure 1: 전체 시스템 아키텍처**
- **위치**: III. The Proposed Scheme 섹션 시작
- **내용**: 
  - Kosmos2+CLIP Hybrid 아키텍처 전체 구조
  - Vision Input → Kosmos2/CLIP → Feature Fusion → LSTM Policy Head → 2D Action Output
  - 각 모듈별 차원 정보 (1024, 2048, 768, 512 → 4352 → 2048 → [linear_x, linear_y])
- **목적**: 제안하는 하이브리드 아키텍처의 전체 구조 이해

### **Figure 2: RoboVLMs 4가지 패러다임 비교**
- **위치**: II. Preliminaries - 1.1 RoboVLMs 아키텍처 분석
- **내용**: 
  - 기존 `robovlms_architectures.md`의 다이어그램 활용
  - 4가지 패러다임 (One-Step-Continuous, Interleaved-Continuous, One-Step-Discrete, Policy-Head-Continuous)
  - 우리 모델이 Policy-Head-Continuous-Action Models에 해당함을 표시
- **목적**: 우리 모델의 위치와 선택 이유 명확화

### **Figure 3: 성능 비교 그래프**
- **위치**: IV. Experimental Results - 3. 성능 평가 결과
- **내용**:
  - 6개 모델의 MAE 성능 막대 그래프
  - 모델별 메모리 사용량 비교
  - 데이터셋 타입별 성능 차이 (원본 vs 증강)
- **목적**: 정량적 성능 비교 시각화

### **Figure 4: 데이터 증강 효과 분석**
- **위치**: IV. Experimental Results - 4. 데이터 증강 효과 분석
- **내용**:
  - 원본 72 에피소드 vs 증강 720 에피소드 성능 비교
  - 증강 기법별 성능 영향 (Horizontal Flip, Action Noise 등)
  - 증강의 역효과 시각화
- **목적**: 데이터 증강의 예상과 다른 결과 강조

## 📈 **추가 고려 그림들 (선택사항)**

### **Figure 5: 실험 환경 설정**
- **위치**: IV. Experimental Results - 1. 실험 환경 및 설정
- **내용**:
  - Jetson Orin NX 하드웨어 구성도
  - ROS2 기반 로봇 제어 시스템 구조
  - 데이터 수집 환경 (8개 시나리오, 72 에피소드)
- **목적**: 실험 환경의 구체적 설정 이해

### **Figure 6: 학습 과정 시각화**
- **위치**: III. The Proposed Scheme - 1.2.4 RoboVLM 학습
- **내용**:
  - 18프레임 시퀀스 처리 과정
  - 4층 LSTM 구조
  - 시간적 구조화 (window-chunk 방식)
- **목적**: 학습 방법론의 구체적 이해

## 🎨 **그림 제작 가이드라인**

### **스타일 통일성**
- **색상**: 일관된 색상 팔레트 사용
- **폰트**: 논문 표준 폰트 (Times New Roman, Arial)
- **크기**: 가독성을 고려한 적절한 크기

### **정보 밀도**
- **최소한의 텍스트**: 핵심 정보만 포함
- **명확한 레이블**: 모든 요소에 적절한 레이블
- **범례**: 색상/기호에 대한 명확한 범례

### **논문 통합**
- **참조**: 본문에서 적절한 Figure 참조
- **설명**: 각 Figure에 대한 충분한 설명
- **연결성**: Figure들 간의 논리적 연결

## 📋 **우선순위**

### **필수 (반드시 포함)**
1. **Figure 1**: 전체 시스템 아키텍처
2. **Figure 2**: RoboVLMs 패러다임 비교
3. **Figure 3**: 성능 비교 그래프

### **권장 (가능하면 포함)**
4. **Figure 4**: 데이터 증강 효과 분석

### **선택 (시간/공간 여유시)**
5. **Figure 5**: 실험 환경 설정
6. **Figure 6**: 학습 과정 시각화

## 🎯 **결론**

최소 3개의 핵심 그림으로 논문의 핵심 내용을 효과적으로 전달할 수 있습니다. 각 그림은 논문의 주요 기여를 시각적으로 뒷받침하는 역할을 하며, 독자의 이해를 크게 향상시킬 것입니다.
