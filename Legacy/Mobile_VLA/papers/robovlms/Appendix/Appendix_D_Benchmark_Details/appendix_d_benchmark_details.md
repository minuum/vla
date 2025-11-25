# 📚 RoboVLMs 논문 Appendix D: BENCHMARK DETAILS 섹션 분석

> **인용**: 논문 "APPENDIX D: BENCHMARK DETAILS" 섹션

## 🎯 **1. 벤치마크 세부사항 개요**

### **벤치마크의 중요성**
> **인용**: "CALVIN [32] is a simulated benchmark for multi-task table-top manipulation." (논문 Appendix D 섹션)

#### **CALVIN 벤치마크**
- **유형**: 시뮬레이션 벤치마크 (Simulated benchmark)
- **목적**: 다중 작업 테이블탑 조작 (Multi-task table-top manipulation)
- **의의**: 로봇 정책의 성능을 체계적으로 평가하는 표준 벤치마크

## 📊 **2. CALVIN 벤치마크 상세 정보**

### **데이터셋 구성**
> **인용**: "It provides 24k human-teleoperated demonstrations annotated with language instruction." (논문 Appendix D 섹션)

#### **데이터 규모**
- **데모 수**: 24,000개 인간 텔레오퍼레이션 데모
- **언어 주석**: 언어 지시사항으로 주석 처리
- **의의**: 대규모 인간 데모 데이터를 통한 학습

### **궤적 특성**
> **인용**: "Each trajectory is less than 64-time steps, which includes 34 pre-defined basic skills" (논문 Appendix D 섹션)

#### **궤적 길이**
- **최대 길이**: 64 타임스텝 미만
- **의의**: 적절한 길이의 궤적으로 효율적 학습

#### **기본 기술 (34개)**
> **인용**: "rotate blue block right, move slider right, lift red block slider, place slider, turn off light bulb, turn off led light, push in drawer, lift blue block drawer, close drawer, lift pink block slider, lift pink block table, move slider left, open drawer, turn on light bulb, rotate blue block left, push blue block left, rotate red block right, turn on led light, push pink block right, push red block left, lift blue block table, place in drawer, rotate red block left, push pink block left, lift stacked blocks, lift blue block slider, push red block right." (논문 Appendix D 섹션)

##### **블록 조작 기술**
- **회전**: rotate blue block right/left, rotate red block right/left
- **이동**: move slider right/left
- **들기**: lift red block slider, lift pink block slider, lift blue block table, lift blue block slider, lift stacked blocks
- **놓기**: place slider, place in drawer
- **밀기**: push blue block left, push pink block right, push red block left, push red block right

##### **조명 제어 기술**
- **켜기**: turn on light bulb, turn on led light
- **끄기**: turn off light bulb, turn off led light

##### **서랍 조작 기술**
- **열기**: open drawer
- **닫기**: close drawer
- **밀어넣기**: push in drawer

### **데이터셋 분할**
> **인용**: "The dataset contains four splits as scene A, B, C, and D." (논문 Appendix D 섹션)

#### **4개 분할**
- **Scene A**: 첫 번째 장면 분할
- **Scene B**: 두 번째 장면 분할
- **Scene C**: 세 번째 장면 분할
- **Scene D**: 네 번째 장면 분할

#### **분할의 목적**
> **인용**: "We train and test VLAs on different training/test splits to fully analyze the capabilities, data and training efficiencies." (논문 Appendix D 섹션)

- **능력 분석**: VLA의 다양한 능력 분석
- **데이터 효율성**: 데이터 효율성 분석
- **훈련 효율성**: 훈련 효율성 분석

### **평가 방법**
> **인용**: "During evaluation, the robot is required to complete a set of 5 consecutive tasks. The metrics are the success rates of finishing these sequential tasks and the average length of achieved tasks." (논문 Appendix D 섹션)

#### **평가 설정**
- **연속 작업**: 5개의 연속된 작업 완료
- **성공률**: 순차 작업 완료 성공률
- **평균 길이**: 달성된 작업의 평균 길이

#### **평가 구현**
> **인용**: "All evaluations are implemented on D split, with 1000 rollouts and 5 consecutive sub-tasks for each rollout." (논문 Appendix D 섹션)

- **평가 분할**: D 분할에서 평가
- **롤아웃 수**: 1000개 롤아웃
- **하위 작업**: 각 롤아웃당 5개의 연속 하위 작업

## 🏗️ **3. SimplerEnv 벤치마크**

### **SimplerEnv 개요**
> **인용**: "SimplerEnv [25] are designed as a suite of real-to-sim environments, which enables evaluating robotic policies in simulation as efficient, scalable, and informative alternative to real-world evaluation." (논문 Appendix D 섹션)

#### **특징**
- **실제-시뮬레이션**: Real-to-sim 환경
- **효율성**: 효율적인 평가 방법
- **확장성**: 확장 가능한 평가
- **정보성**: 정보가 풍부한 평가

### **비교 가능한 환경**
> **인용**: "SimplerEnv created a comparable arena for benchmarking robotics policies on private real-world setups as Google [6, 7] and BridgeData V2 [45]." (논문 Appendix D 섹션)

#### **비교 대상**
- **Google Robot**: Google의 실제 로봇 설정
- **BridgeData V2**: BridgeData V2 데이터셋
- **의의**: 실제 환경과 비교 가능한 시뮬레이션 환경

## 🤖 **4. Google Robot 설정 작업**

### **pick coke can**
> **인용**: "The task assigned to the robot is to pick up an empty Coke can from the table and lift it. Under the standard configuration, the environment is kept free of any distracting elements." (논문 Appendix D 섹션)

#### **작업 설명**
- **목표**: 빈 콜라 캔을 테이블에서 집어서 들어올리기
- **환경**: 방해 요소가 없는 표준 구성
- **난이도**: 기본적인 집기 작업

#### **실험 설정**
> **인용**: "The Coke can is arranged in three distinct positions: lying flat horizontally, lying flat vertically, and standing upright. For each of these positions, the can is placed at 25 specific grid points within a defined rectangular area on the table. This setup results in 25 experiments per position, totaling 75 trials across all orientations." (논문 Appendix D 섹션)

##### **위치 설정**
- **수평 누워있음**: lying flat horizontally
- **수직 누워있음**: lying flat vertically
- **세워져 있음**: standing upright

##### **그리드 배치**
- **그리드 포인트**: 25개 특정 그리드 포인트
- **영역**: 정의된 사각형 영역 내
- **총 실험**: 위치당 25개 실험, 총 75개 시행

### **move {obj1} near {obj2}**
> **인용**: "In the experiment, a set of three objects was arranged on the table in a triangular formation. For each trial, one object was assigned the role of the source, another was designated as the target, and the third served as a distractor." (논문 Appendix D 섹션)

#### **작업 설명**
- **목표**: 한 객체를 다른 객체 근처로 이동
- **형태**: 삼각형 배치
- **역할**: 소스, 타겟, 방해물

#### **실험 설정**
> **인용**: "This setup resulted in six distinct trials for each triplet and triangular configuration. From a total of eight objects—blue plastic bottle, Pepsi can, orange, 7up can, apple, sponge, Coke can, and Redbull can—five triplets were randomly selected. Additionally, two triangular patterns, upright and inverted, were employed. This design produced a total of 60 trials." (논문 Appendix D 섹션)

##### **객체 구성**
- **총 객체**: 8개 (blue plastic bottle, Pepsi can, orange, 7up can, apple, sponge, Coke can, Redbull can)
- **선택**: 5개 삼중체 무작위 선택
- **패턴**: 직립 및 뒤집힌 삼각형 패턴

##### **실험 수**
- **삼중체당**: 6개 구별되는 시행
- **총 시행**: 60개 시행

### **(open/close) (top/middle/bottom) drawer**
> **인용**: "In this setup, the robot is placed facing a cabinet equipped with three drawers and tasked with opening or closing a specific drawer. This experiment evaluates the robot's capability to handle articulated objects." (논문 Appendix D 섹션)

#### **작업 설명**
- **목표**: 특정 서랍 열기/닫기
- **서랍**: 3개 서랍 (상단, 중간, 하단)
- **평가**: 관절 객체 처리 능력

#### **실험 설정**
> **인용**: "The robot is positioned at nine distinct locations on a predefined grid within a rectangular area on the floor. With three drawers and two possible actions (opening or closing), the setup results in a total of 54 trials." (논문 Appendix D 섹션)

##### **로봇 위치**
- **위치**: 9개 구별되는 위치
- **그리드**: 사각형 영역 내 사전 정의된 그리드
- **영역**: 바닥의 사각형 영역

##### **실험 수**
- **서랍**: 3개
- **동작**: 2개 (열기/닫기)
- **총 시행**: 54개 시행

### **open top drawer; place apple into top drawer**
> **인용**: "In this experiment, the robot is tasked with opening the top drawer and transferring an apple from the surface of the cabinet into the drawer. This setup evaluates the robot's ability to execute tasks that require multiple sequential actions." (논문 Appendix D 섹션)

#### **작업 설명**
- **목표**: 상단 서랍 열기 + 사과를 서랍에 넣기
- **평가**: 다중 순차 동작 실행 능력
- **복잡성**: 2단계 순차 작업

#### **실험 설정**
> **인용**: "The robot is positioned in three distinct locations on the floor, while the apple is placed at nine specific grid points on the cabinet surface, resulting in a total of 27 trials. At the start, the robot operates under the instruction to open the top drawer. Once the robot either signals task completion with a 'terminate' token or reaches the midpoint of the allotted time, the instruction transitions to directing the robot to place the apple into the drawer." (논문 Appendix D 섹션)

##### **위치 설정**
- **로봇 위치**: 3개 구별되는 위치
- **사과 위치**: 캐비닛 표면의 9개 특정 그리드 포인트
- **총 시행**: 27개 시행

##### **작업 순서**
- **1단계**: 상단 서랍 열기
- **완료 신호**: "terminate" 토큰 또는 시간 중점 도달
- **2단계**: 사과를 서랍에 넣기

## 🔧 **5. WidowX + Bridge 설정 작업**

### **put the spoon on the towel**
> **인용**: "In this setup, the spoon is positioned at one corner of a square on the tabletop, with the towel placed at a different corner. The square has sides measuring 15 cm in length." (논문 Appendix D 섹션)

#### **작업 설명**
- **목표**: 숟가락을 수건 위에 놓기
- **배치**: 사각형의 서로 다른 모서리
- **크기**: 15cm 길이의 사각형

#### **실험 설정**
> **인용**: "The orientation of the spoon alternates between horizontal and vertical, requiring the robot to adjust the orientation of its gripper accordingly. This configuration results in a total of 24 trials." (논문 Appendix D 섹션)

##### **방향 설정**
- **수평**: horizontal orientation
- **수직**: vertical orientation
- **그리퍼 조정**: 방향에 따른 그리퍼 조정 필요

##### **실험 수**
- **총 시행**: 24개 시행

### **put carrot on plate**
> **인용**: "This setup is similar to put the spoon on the towel, but the spoon is replaced with a carrot and the towel is substituted with a plate." (논문 Appendix D 섹션)

#### **작업 설명**
- **목표**: 당근을 접시 위에 놓기
- **유사성**: 숟가락-수건 작업과 유사
- **차이점**: 객체 변경 (숟가락→당근, 수건→접시)

### **stack the green block on the yellow block**
> **인용**: "In this experiment, a green block is positioned at one corner of a square on the tabletop, while a yellow block is placed at a different corner. Both blocks measure 3 cm in size." (논문 Appendix D 섹션)

#### **작업 설명**
- **목표**: 녹색 블록을 노란색 블록 위에 쌓기
- **배치**: 사각형의 서로 다른 모서리
- **크기**: 3cm 크기의 블록

#### **실험 설정**
> **인용**: "Two square configurations with 10 cm and 20 cm side lengths are used. This setup results in a total of 24 trials." (논문 Appendix D 섹션)

##### **사각형 구성**
- **10cm**: 10cm 변 길이
- **20cm**: 20cm 변 길이
- **총 시행**: 24개 시행

### **put eggplant into yellow basket**
> **인용**: "An eggplant is positioned randomly within the right basin of a sink, while a yellow basket is placed in the left basin. The eggplant's placement varies in both location and orientation but is carefully arranged to remain easily graspable, avoiding proximity to the sink's edges." (논문 Appendix D 섹션)

#### **작업 설명**
- **목표**: 가지를 노란색 바구니에 넣기
- **위치**: 싱크의 오른쪽/왼쪽 대야
- **난이도**: 위치와 방향의 다양성

#### **실험 설정**
> **인용**: "A total of 24 trials are conducted under this setup." (논문 Appendix D 섹션)

##### **배치 특징**
- **가지**: 오른쪽 대야 내 무작위 위치
- **바구니**: 왼쪽 대야에 배치
- **조정**: 쉽게 잡을 수 있도록 조정
- **경계**: 싱크 가장자리 근처 피하기

##### **실험 수**
- **총 시행**: 24개 시행

## 📈 **6. 벤치마크 비교 분석**

### **CALVIN vs SimplerEnv**

| 특징 | CALVIN | SimplerEnv |
|------|--------|------------|
| **유형** | 시뮬레이션 | 실제-시뮬레이션 |
| **데이터** | 24K 데모 | 실제 환경 기반 |
| **작업** | 34개 기본 기술 | 다양한 실제 작업 |
| **평가** | 연속 작업 | 개별 작업 |

### **작업 복잡도 분석**

#### **기본 작업 (CALVIN)**
- **단순 조작**: 블록 회전, 이동, 들기
- **기본 제어**: 조명 켜기/끄기, 서랍 열기/닫기
- **표준화**: 34개 사전 정의된 기술

#### **복합 작업 (SimplerEnv)**
- **다단계 작업**: 서랍 열기 + 사과 넣기
- **객체 조작**: 다양한 객체와 도구 사용
- **공간 인식**: 정확한 위치와 방향 조정

## 🎯 **7. 벤치마크의 의의**

### **평가의 중요성**
1. **표준화**: 로봇 정책 평가의 표준화
2. **비교**: 다양한 방법론의 공정한 비교
3. **진전**: 연구 진전의 객관적 측정

### **다양성의 가치**
1. **기본 기술**: CALVIN의 34개 기본 기술
2. **실제 작업**: SimplerEnv의 실제 작업
3. **복합 능력**: 다단계 순차 작업

### **연구 생태계**
1. **표준**: 연구 커뮤니티의 표준 벤치마크
2. **재현성**: 재현 가능한 평가 환경
3. **확장성**: 새로운 작업과 환경으로의 확장

## 🚀 **8. 결론**

### **벤치마크 세부사항의 핵심**
1. **체계성**: 체계적인 벤치마크 설계
2. **다양성**: 다양한 작업과 환경
3. **표준화**: 표준화된 평가 방법

### **연구의 의의**
1. **평가**: 객관적이고 공정한 평가
2. **비교**: 다양한 방법론의 비교
3. **진전**: 연구 진전의 측정

### **미래 방향**
1. **확장**: 새로운 작업과 환경 추가
2. **표준화**: 더 넓은 커뮤니티의 표준화
3. **혁신**: 새로운 평가 방법론 개발

---

*분석 작성일: 2024년 12월*  
*원본 논문: "Towards Generalist Robot Policies: What Matters in Building Vision-Language-Action Models"*