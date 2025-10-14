# RoboVLMs Appendix Analysis

## 구현 세부사항

### 하이퍼파라미터 설정

#### CALVIN 성능 실험
- **배치 크기**: 128
- **워밍업**: 0.25 epoch
- **스케줄러**: Constant
- **옵티마이저**: AdamW
- **학습률**: 1e-4
- **총 에포크**: 5

#### SimplerEnv 성능 실험
- **배치 크기**: 128
- **워밍업**: 5K iterations
- **스케줄러**: Constant
- **옵티마이저**: AdamW
- **학습률**: 1e-4
- **총 반복**: 50K

#### 실제 로봇 실험
- **배치 크기**: 128
- **워밍업**: 0.25 epoch
- **스케줄러**: Constant
- **옵티마이저**: AdamW
- **학습률**: 1e-4
- **총 에포크**: 5

### 체크포인트 선택 전략

#### 문제점
- 로봇 정책의 성능이 오프라인 평가 지표(검증 손실)에 완전히 의존하지 않음
- 장기간 롤아웃에서의 복합 오류로 인한 어려움

#### 해결책
- **고정 에포크/반복 수** 사용
- **CALVIN**: 5 에포크, 최종 모델 성능 보고
- **SimplerEnv**: 100K 반복, 10K 반복 간격으로 최고 성능 모델 선택
- **실제 로봇**: 5 에포크, 최종 모델 성능 보고

## 벤치마크 상세 정보

### CALVIN 벤치마크

#### 데이터셋 구성
- **총 시연**: 24K 인간 텔레오퍼레이션 시연
- **언어 지시**: 모든 시연에 언어 지시 포함
- **궤적 길이**: 64 시간 단계 이하
- **기본 기술**: 34개 사전 정의된 기본 기술

#### 34개 기본 기술 목록
1. rotate blue block right
2. move slider right
3. lift red block slider
4. place slider
5. turn off light bulb
6. turn off led light
7. push in drawer
8. lift blue block drawer
9. close drawer
10. lift pink block slider
11. lift pink block table
12. move slider left
13. open drawer
14. turn on light bulb
15. rotate blue block left
16. turn on led light
17. push pink block right
18. push red block left
19. lift blue block table
20. place in drawer
21. rotate red block left
22. push pink block left
23. lift stacked blocks
24. lift blue block slider
25. push blue block right

#### 분할 설정
- **A, B, C, D**: 장면 설정별 분할
- **훈련/테스트 분할**: 다양한 조합으로 일반화 능력 평가
- **평가**: D 분할에서 1000 롤아웃, 5개 연속 하위 작업

### SimplerEnv 벤치마크

#### WidowX+Bridge 설정

##### Put Spoon on Towel
- **설정**: 15cm x 15cm 정사각형 테이블
- **스푼 위치**: 정사각형 한 모서리
- **타월 위치**: 다른 모서리
- **방향**: 수평/수직 교대
- **총 시행**: 24회

##### Put Carrot on Plate
- **설정**: Put Spoon on Towel과 유사
- **스푼 → 당근**: 스푼을 당근으로 교체
- **타월 → 접시**: 타월을 접시로 교체

##### Stack Green Block on Yellow Block
- **설정**: 10cm, 20cm 정사각형 구성
- **녹색 블록**: 한 모서리
- **노란색 블록**: 다른 모서리
- **블록 크기**: 3cm
- **총 시행**: 24회

##### Put Eggplant in Yellow Basket
- **설정**: 싱크의 좌우 대야
- **가지**: 우측 대야에 무작위 위치
- **노란 바구니**: 좌측 대야
- **총 시행**: 24회

#### Google Robot 설정

##### Pick Coke Can
- **환경**: 방해 요소 없는 표준 구성
- **캔 위치**: 25개 그리드 포인트
- **방향**: 수평(누워), 수직(누워), 서 있는 상태
- **총 시행**: 75회 (25 x 3 방향)

##### Move {obj1} near {obj2}
- **설정**: 3개 객체를 삼각형으로 배치
- **역할**: 소스, 타겟, 방해 요소
- **객체**: 8개 중 5개 삼중항 무작위 선택
- **패턴**: 정삼각형, 역삼각형
- **총 시행**: 60회

##### Open/Close Drawer
- **설치**: 3개 서랍이 있는 캐비닛
- **로봇 위치**: 9개 그리드 위치
- **동작**: 열기/닫기
- **총 시행**: 54회

##### Open Top Drawer & Place Apple
- **작업**: 서랍 열기 + 사과 이동
- **로봇 위치**: 3개 위치
- **사과 위치**: 캐비닛 표면 9개 그리드 포인트
- **총 시행**: 27회

### 실제 로봇 플랫폼

#### 하드웨어 구성
- **로봇 팔**: Kinova Gen-3 (7-DoF)
- **그리퍼**: Robotiq 2F-85 병렬 턱 그리퍼
- **카메라**: 
  - Kinect Azure (정적 카메라)
  - RealSense D435i (손목 카메라)
- **작업 공간**: 55cm x 24cm 테이블
- **객체**: 40개 이상의 다양한 객체

#### 평가 설정

##### Simple 설정
- **목적**: 훈련 데이터 분포에 대한 모델 적합성 평가
- **특징**: 훈련 데이터와 유사한 장면

##### Unseen Distractor 설정
- **목적**: 이전에 보지 못한 방해 객체 도입
- **특징**: 조작 객체는 훈련 데이터에 포함

##### Unseen Background 설정
- **목적**: 배경 변경
- **방법**: 훈련 데이터에 없는 새로운 테이블보 추가
- **특징**: 색상과 패턴이 다른 테이블보

##### Unseen Object 설정
- **목적**: 훈련 데이터에 없는 객체 조작
- **특징**: Unseen Distractor와 동일한 객체 사용

##### Novel Skill Description 설정
- **목적**: 새로운 동사로 작업 지시 변경
- **방법**: GPT-4로 동사의 동의어 3개 생성
- **예시**: "press" → "hit", "pick up" → "take"

## 성능 분석

### CALVIN 상세 성능

#### 일반화 및 데이터 효율성
- **VL 사전 훈련 효과**: 4.49 vs 2.70 Avg. Len.
- **데이터 스케일 효과**: 5x 데이터로 4.51 Avg. Len.
- **모델 크기 효과**: 9B 모델이 3B 대비 향상

#### 다양한 백본 성능
- **KosMos**: 최고 성능
- **Flamingo**: 중간 성능
- **LLaVA**: Perceiver Resampler 추가 시 향상

### SimplerEnv 상세 성능

#### WidowX+Bridge 결과
| 방법 | Put Spoon | Put Carrot | Stack Block | Put Eggplant |
|------|-----------|------------|-------------|--------------|
| Cross-Emb Pre-Train | 0.375 | 0.208 | 0.333 | 0.250 |
| In-domain Full Finetune | 0.542 | 0.292 | 0.250 | 0.250 |
| Post Train | 0.708 | 0.458 | 0.333 | 0.208 |

#### Google Robot 결과
| 방법 | Pick Coke | Move Near | Open/Close | Open & Place |
|------|-----------|-----------|-------------|--------------|
| Cross-Emb Pre-Train | 0.850 | 0.430 | 0.900 | 0.727 |
| In-domain Full Finetune | 0.920 | 0.810 | 0.980 | 0.903 |
| Post Train | 0.940 | 0.470 | 0.910 | 0.773 |

## 롤아웃 예시

### 성공 사례
- **Open Drawer**: 정확한 서랍 위치 인식 및 조작
- **Pickup Eggplant**: 적절한 그리퍼 각도로 조작
- **Press Toaster**: 정확한 버튼 위치 인식

### 실패 사례
- **위치 오차**: 목표 객체에서 약간 벗어난 위치
- **각도 오차**: 그리퍼 각도 조정 실패
- **타이밍 오차**: 동작 순서 오류

## 기술적 세부사항

### 모델 아키텍처
- **VLM 백본**: 8개 다양한 백본 지원
- **VLA 구조**: 4가지 구조 구현
- **정책 헤드**: RNN, Transformer, Diffusion 모델 지원

### 훈련 전략
- **사전 훈련**: Vision-Language 데이터
- **파인튜닝**: 로봇 조작 데이터
- **Post-training**: Cross-embodiment 데이터 활용

### 평가 방법
- **시뮬레이션**: CALVIN, SimplerEnv
- **실제 로봇**: 20개 작업, 5가지 설정
- **메트릭**: 성공률, 평균 달성 작업 수
