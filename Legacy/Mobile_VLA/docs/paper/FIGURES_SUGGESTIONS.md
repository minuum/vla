# 📊 논문 그림/Figure 제안사항

## 🎯 **최소한의 필수 그림들**

### **Figure 1: 전체 시스템 아키텍처**
- **위치**: III. The Proposed Scheme - 3.3 구현 사항
- **내용**: 
  - Kosmos2+CLIP Hybrid 아키텍처 전체 구조
  - Vision Input → Kosmos2/CLIP → Feature Fusion → LSTM Policy Head → 2D Action Output
  - 각 모듈별 차원 정보 (1024, 2048, 768, 512 → 4352 → 2048 → [linear_x, linear_y])
  - 4층 LSTM 구조 (4096 hidden size)
- **목적**: 제안하는 하이브리드 아키텍처의 전체 구조 이해

### **Figure 2: RoboVLMs 4가지 패러다임 비교**
- **위치**: II. Preliminaries - 1.1 RoboVLMs 아키텍처 분석
- **내용**: 
  - 4가지 패러다임 분류 다이어그램
  - One-Step-Continuous, Interleaved-Continuous, One-Step-Discrete, Policy-Head-Continuous
  - 우리 모델이 Policy-Head-Continuous-Action Models에 해당함을 강조 표시
  - 7-DoF 로봇 팔 제어 → 2-DoF 모바일 로봇 내비게이션 적용 표시
- **목적**: 우리 모델의 위치와 선택 이유 명확화

### **Figure 3: 성능 비교 그래프**
- **위치**: IV. Experimental Results - 3. 성능 평가 결과
- **내용**:
  - 6개 모델의 MAE 성능 막대 그래프
    - Kosmos2+CLIP Hybrid: 0.212
    - Pure Kosmos2: 0.247
    - Simple CLIP: 0.451
    - CLIP with LSTM: 0.456
    - Original CLIP: 0.494
    - Original CLIP (증강): 0.672
  - 모델별 메모리 사용량 비교
    - CLIP 기반 모델들: 1.7GB
    - Pure Kosmos2: 6.8GB
    - Kosmos2+CLIP Hybrid: 7.4GB
- **목적**: 정량적 성능 비교 시각화

### **Figure 4: 데이터 증강 효과 분석**
- **위치**: IV. Experimental Results - 4. 데이터 증강 효과 분석
- **내용**:
  - 원본 72 에피소드 vs 증강 720 에피소드 성능 비교
  - 증강 기법별 성능 영향 (Horizontal Flip, Action Noise, Forward/Backward Flip, Speed Variation, Start/Stop Pattern)
  - 증강의 역효과 시각화 (MAE 0.494 → 0.672)
- **목적**: 데이터 증강의 예상과 다른 결과 강조

## 📈 **추가 고려 그림들 (선택사항)**

### **Figure 5: 실험 환경 설정**
- **위치**: IV. Experimental Results - 1. 실험 환경 및 설정
- **내용**:
  - Jetson Orin NX 16GB 하드웨어 구성도
    - ARM Cortex-A78AE 8-core CPU
    - NVIDIA Ampere 1024-core GPU
    - 16GB LPDDR5 메모리
    - 64GB eMMC 저장공간
  - 소프트웨어 환경 구성
    - Ubuntu 22.04 LTS, Python 3.10, PyTorch 2.0+, ROS2 Humble, CUDA 11.8
  - 데이터 수집 환경 (8개 시나리오, 72 에피소드)
- **목적**: 실험 환경의 구체적 설정 이해

### **Figure 6: 학습 과정 시각화**
- **위치**: III. The Proposed Scheme - 1.2.4 RoboVLM 학습
- **내용**:
  - 18프레임 시퀀스 처리 과정
  - 4층 LSTM 구조 (4096 hidden size)
  - 시간적 구조화 (window-chunk 방식)
  - 학습 파라미터: 배치 크기 2, Adam 옵티마이저, 10-15 에포크
- **목적**: 학습 방법론의 구체적 이해

### **Figure 7: 시간적 구조화 시퀀스 다이어그램**
- **위치**: III. The Proposed Scheme - 1.2.4 시간적 구조화
- **내용**:
  - 윈도우-청크 분할 구조
  - 과거 8프레임 관찰 → 미래 10프레임 예측
  - 시간 순서별 동작 시뮬레이션 (t=0~12)
  - 과거/미래 정보 유출 차단 메커니즘
- **목적**: 시간적 구조화 방법의 구체적 이해

### **Figure 8: 하이브리드 특징 융합 구조**
- **위치**: III. The Proposed Scheme - 3.4 학습 수행
- **내용**:
  - Kosmos-2 Vision 특징 (1024차원)
  - Kosmos-2 Text 특징 (2048차원)
  - CLIP Vision 특징 (768차원)
  - CLIP Text 특징 (512차원)
  - Concatenation → 4352차원 → Linear Layer → 2048차원 → LSTM
- **목적**: 하이브리드 아키텍처의 특징 융합 과정 이해

## 🎨 **그림 제작 가이드라인**

### **스타일 통일성**
- **색상**: 일관된 색상 팔레트 사용 (논문 표준)
- **폰트**: 논문 표준 폰트 (Times New Roman, Arial)
- **크기**: 가독성을 고려한 적절한 크기
- **해상도**: 300 DPI 이상 (인쇄 품질)

### **정보 밀도**
- **최소한의 텍스트**: 핵심 정보만 포함
- **명확한 레이블**: 모든 요소에 적절한 레이블
- **범례**: 색상/기호에 대한 명확한 범례
- **수치 표시**: 정확한 수치 정보 포함

### **논문 통합**
- **참조**: 본문에서 적절한 Figure 참조
- **설명**: 각 Figure에 대한 충분한 설명
- **연결성**: Figure들 간의 논리적 연결
- **일관성**: 용어와 표기법의 일관성 유지

## 📋 **우선순위**

### **필수 (반드시 포함)**
1. **Figure 1**: 전체 시스템 아키텍처
2. **Figure 2**: RoboVLMs 패러다임 비교
3. **Figure 3**: 성능 비교 그래프

### **권장 (가능하면 포함)**
4. **Figure 4**: 데이터 증강 효과 분석
5. **Figure 7**: 시간적 구조화 시퀀스 다이어그램

### **선택 (시간/공간 여유시)**
6. **Figure 5**: 실험 환경 설정
7. **Figure 6**: 학습 과정 시각화
8. **Figure 8**: 하이브리드 특징 융합 구조

## 🎯 **결론**

최소 3개의 핵심 그림으로 논문의 핵심 내용을 효과적으로 전달할 수 있습니다. 각 그림은 논문의 주요 기여를 시각적으로 뒷받침하는 역할을 하며, 독자의 이해를 크게 향상시킬 것입니다. 특히 RoboVLMs 패러다임 적용과 하이브리드 아키텍처의 성능 향상을 시각적으로 명확하게 보여주는 것이 중요합니다.
