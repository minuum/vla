# RoboVLMs 논문 Appendix K: ROLLOUT EXAMPLES IN REAL-WORLD EXPERIMENTS 섹션 분석

> **인용**: 논문 "APPENDIX K: ROLLOUT EXAMPLES IN REAL-WORLD EXPERIMENTS" 섹션

## 1. 실제 환경 실험 롤아웃 예시 개요

### Fig. 15: 실제 환경 작업의 실험 설정
> **인용**: "Fig. 15: This figure shows the experimental setups for some real-world tasks." (논문 Fig. 15)

#### Fig. 15의 목적
- **실험 설정 시각화**: 실제 환경에서의 다양한 실험 조건 시각화
- **일반화 능력 평가**: 로봇 정책의 일반화 능력 평가를 위한 다양한 조건
- **작업별 롤아웃**: 20가지 작업에 대한 5가지 롤아웃 조건
- **환경 다양성**: 다양한 환경 조건에서의 로봇 성능 검증

### 실험 설정의 구성
- **6개 열**: Various Skills & Objects, Basic, Unseen Distractor, Unseen Background, Unseen Target Object, Novel Skill Description
- **15개 행**: 다양한 작업과 대상 객체
- **로봇 시스템**: Kinova Gen3 로봇 암과 Robotiq 2F-85 그리퍼
- **평가 조건**: 5가지 롤아웃 조건

## 2. 6가지 실험 조건 분석

### 1. Various Skills & Objects (다양한 기술과 객체)
- **작업 유형**: 15가지 다양한 조작 작업
- **대상 객체**: 칼, 오이, 가지, 녹색 병, 빨간 컵, 녹색 컵, 만다린, 감자 등
- **기술 유형**: 집기, 토스터 스위치 누르기, 오븐 열기/닫기, 양념 가루 붓기/집기, 서랍 열기 등
- **초기 상태**: 로봇 팔이 작업 대상 객체 근처에 위치

### 2. Basic (기본 설정)
- **환경 조건**: 표준 환경에서의 작업 수행
- **로봇 동작**: 각 작업에 대한 기본적인 로봇 동작
- **성공 사례**: 칼 집기, 오븐 손잡이 당기기, 서랍 손잡이 잡기 등
- **기준 성능**: 다른 조건과 비교하기 위한 기준 성능

### 3. Unseen Distractor (미확인 방해물)
- **방해물 추가**: 기존 훈련 데이터에 없던 새로운 방해물 추가
- **시각적 표시**: 대상 객체와 방해물을 노란색 경계 상자로 표시
- **예시**: 칼 옆에 작은 흰색 물체, 오이 옆에 노란색 블록
- **일반화 테스트**: 방해물이 있는 환경에서의 작업 수행 능력

### 4. Unseen Background (미확인 배경)
- **배경 변경**: 훈련 데이터에 없던 새로운 배경 사용
- **배경 패턴**: 흰색 바탕 대신 흰색과 분홍색 패턴의 식탁보
- **시각적 표시**: 대상 객체를 노란색 경계 상자로 표시
- **일반화 테스트**: 새로운 배경에서의 작업 수행 능력

### 5. Unseen Target Object (미확인 대상 객체)
- **대상 객체 변경**: 훈련 데이터에 없던 새로운 대상 객체 사용
- **시각적 표시**: 새로운 대상 객체를 노란색 경계 상자로 표시
- **예시**: 칼 대신 노란색 병, 가지 대신 흰색 병
- **제한 사항**: 일부 작업은 적절한 대체 미확인 객체가 부족하여 제외
- **일반화 테스트**: 새로운 객체에 대한 작업 수행 능력

### 6. Novel Skill Description (새로운 기술 설명)
- **지시어 변경**: 원래 작업 지시를 새로운 동의어로 대체
- **동의어 예시**: "Pick up" → "Take" 또는 "Grab", "Press" → "Hit", "Pour" → "Sprinkle"
- **예시**: "Pick up the knife" → "Take the knife"
- **일반화 테스트**: 새로운 지시어에 대한 작업 수행 능력

## 3. 15가지 작업별 분석

### 물체 조작 작업 (8개)

#### 1. Pick up Knife (칼 집기)
- **대상 객체**: 칼
- **작업 유형**: 집기 작업
- **일반화 조건**: Unseen Distractor, Unseen Background, Unseen Target Object, Novel Skill Description

#### 2. Pick up Cucumber (오이 집기)
- **대상 객체**: 오이
- **작업 유형**: 집기 작업
- **일반화 조건**: Unseen Distractor, Unseen Background, Unseen Target Object, Novel Skill Description

#### 3. Pick up Eggplant (가지 집기)
- **대상 객체**: 가지
- **작업 유형**: 집기 작업
- **일반화 조건**: Unseen Distractor, Unseen Background, Unseen Target Object, Novel Skill Description

#### 4. Pick up Green Bottle (녹색 병 집기)
- **대상 객체**: 녹색 병
- **작업 유형**: 집기 작업
- **일반화 조건**: Unseen Distractor, Unseen Background, Unseen Target Object, Novel Skill Description

#### 5. Pick up Red Cup (빨간 컵 집기)
- **대상 객체**: 빨간 컵
- **작업 유형**: 집기 작업
- **일반화 조건**: Unseen Distractor, Unseen Background, Unseen Target Object, Novel Skill Description

#### 6. Pick up Green Cup (녹색 컵 집기)
- **대상 객체**: 녹색 컵
- **작업 유형**: 집기 작업
- **일반화 조건**: Unseen Distractor, Unseen Background, Unseen Target Object, Novel Skill Description

#### 7. Pick up Mandarin (만다린 집기)
- **대상 객체**: 만다린
- **작업 유형**: 집기 작업
- **일반화 조건**: Unseen Distractor, Unseen Background, Unseen Target Object, Novel Skill Description

#### 8. Pick up Potato (감자 집기)
- **대상 객체**: 감자
- **작업 유형**: 집기 작업
- **일반화 조건**: Unseen Distractor, Unseen Background, Unseen Target Object, Novel Skill Description

### 가전제품 조작 작업 (3개)

#### 9. Press Toaster Switch (토스터 스위치 누르기)
- **대상 객체**: 토스터 스위치
- **작업 유형**: 스위치 누르기
- **일반화 조건**: Unseen Distractor, Unseen Background, Novel Skill Description
- **제한 사항**: Unseen Target Object 설정 제외

#### 10. Open/Close Oven (오븐 열기/닫기)
- **대상 객체**: 오븐 손잡이
- **작업 유형**: 열기/닫기 작업
- **일반화 조건**: Unseen Distractor, Unseen Background, Novel Skill Description
- **제한 사항**: Unseen Target Object 설정 제외

#### 11. Pour/Scoop Seasoning Powder (양념 가루 붓기/집기)
- **대상 객체**: 양념 가루
- **작업 유형**: 붓기/집기 작업
- **일반화 조건**: Unseen Distractor, Unseen Background, Novel Skill Description
- **제한 사항**: Unseen Target Object 설정 제외

### 서랍 조작 작업 (4개)

#### 12. Open Top Drawer (위 서랍 열기)
- **대상 객체**: 위 서랍 손잡이
- **작업 유형**: 서랍 열기
- **일반화 조건**: Unseen Distractor, Unseen Background, Novel Skill Description
- **제한 사항**: Unseen Target Object 설정 제외

#### 13. Open Middle Drawer (중간 서랍 열기)
- **대상 객체**: 중간 서랍 손잡이
- **작업 유형**: 서랍 열기
- **일반화 조건**: Unseen Distractor, Unseen Background, Novel Skill Description
- **제한 사항**: Unseen Target Object 설정 제외

#### 14. Open Bottom Drawer (아래 서랍 열기)
- **대상 객체**: 아래 서랍 손잡이
- **작업 유형**: 서랍 열기
- **일반화 조건**: Unseen Distractor, Unseen Background, Novel Skill Description
- **제한 사항**: Unseen Target Object 설정 제외

#### 15. Close Drawer (서랍 닫기)
- **대상 객체**: 서랍 손잡이
- **작업 유형**: 서랍 닫기
- **일반화 조건**: Unseen Distractor, Unseen Background, Novel Skill Description
- **제한 사항**: Unseen Target Object 설정 제외

## 4. 실험 설정의 특징

### 로봇 시스템
- **로봇 암**: Kinova Gen3 로봇 암
- **그리퍼**: Robotiq 2F-85 그리퍼
- **카메라**: 헤드 카메라와 손목 카메라
- **작업 공간**: 55cm x 24cm 테이블

### 평가 방법론
- **총 20개 작업**: 다양한 조작 작업
- **5번의 롤아웃**: 각 작업당 5번의 롤아웃
- **5가지 조건**: Basic, Unseen Distractor, Unseen Background, Unseen Target Object, Novel Skill Description
- **성공률 측정**: 각 조건별 성공률 측정

### 일반화 능력 평가
- **방해물 적응**: 새로운 방해물이 있는 환경에서의 작업 수행
- **배경 적응**: 새로운 배경에서의 작업 수행
- **객체 적응**: 새로운 객체에 대한 작업 수행
- **지시어 적응**: 새로운 지시어에 대한 작업 수행

## 5. 실험 결과의 시사점

### RoboVLMs의 일반화 능력
- **다양한 조건**: 5가지 다양한 조건에서의 작업 수행
- **일관된 성능**: 다양한 조건에서 일관된 성능 유지
- **적응 능력**: 새로운 환경과 객체에 대한 적응 능력

### 작업별 성능 특성
- **물체 조작**: 다양한 객체에 대한 조작 능력
- **가전제품 조작**: 복잡한 가전제품 조작 능력
- **서랍 조작**: 정밀한 서랍 조작 능력

### 일반화 조건별 성능
- **Basic**: 기준 성능
- **Unseen Distractor**: 방해물이 있는 환경에서의 성능
- **Unseen Background**: 새로운 배경에서의 성능
- **Unseen Target Object**: 새로운 객체에 대한 성능
- **Novel Skill Description**: 새로운 지시어에 대한 성능

## 6. 실제 환경 실험의 상세 분석

### Fig. 14: SimplerEnv 작업에 대한 질적 결과
> **인용**: "Fig. 14: Qualitative results for SimplerEnv tasks." (논문 Fig. 14)

#### Fig. 14의 구성
- **12개 작업**: 다양한 조작 작업의 순차적 실행 과정
- **4개 프레임**: 각 작업당 2행 2열의 순차적 프레임
- **시뮬레이션 환경**: SimplerEnv 시뮬레이션 환경에서의 성능

#### 주요 작업 분석
1. **물체 배치 작업**: Put Carrot on Plate, Place Eggplant in Bucket
2. **물체 조작 작업**: Put Spoon on Towel, Stack Green Block on Yellow
3. **서랍 조작 작업**: Open/Close The Bottom/Top/Middle Drawer
4. **파지 및 이동 작업**: Grasp Coke Can, Move Near

#### 성능 특성
- **정확한 조작**: 각 작업의 순차적 실행 과정
- **안정적 파지**: 다양한 물체의 안정적 파지
- **정밀한 배치**: 목표 위치에의 정확한 배치

### Fig. 16: 기본 설정에서의 실제 환경 실험 질적 결과
> **인용**: "Fig. 16: Qualitative results for basic setting in real-world experiments." (논문 Fig. 16)

#### Fig. 16의 구성
- **7개 작업**: 실제 환경에서의 기본 조작 작업
- **카메라 시점**: Side Camera와 Wrist Camera의 이중 시점
- **8개 프레임**: 각 카메라별 8개 프레임의 순차적 과정

#### 주요 작업 분석
1. **Open The Drawer**: 서랍 열기 작업
2. **Press the toaster switch**: 토스터 스위치 누르기
3. **Pick up the knife**: 칼 집어 서랍에 넣기
4. **Pick up the eggplant**: 가지 집어 테이블에 놓기
5. **Pick up the red mug**: 빨간 머그컵 집어 테이블에 놓기
6. **Pick up the cucumber**: 오이 집어 도마에 놓기
7. **Pick up the potato**: 감자 집어 도마에 놓기

#### 성능 특성
- **이중 시점**: Side Camera와 Wrist Camera의 상호 보완적 시각화
- **정확한 조작**: 각 작업의 성공적 완료
- **안정적 파지**: 다양한 물체의 안정적 파지와 이동

### Fig. 17: 미확인 배경에서의 질적 결과
> **인용**: "Fig. 17: Qualitative results for unseen background." (논문 Fig. 17)

#### Fig. 17의 구성
- **7개 작업**: 미확인 배경에서의 조작 작업
- **배경 다양성**: 다양한 패턴의 식탁보와 배경
- **일반화 테스트**: 새로운 배경에서의 작업 수행 능력

#### 주요 작업 분석
1. **Pick up the red mug**: 빨간 머그컵 집어 테이블에 놓기
2. **Pick up the mandarin**: 만다린 집어 빨간 접시에 놓기
3. **Open the drawer**: 서랍 열기
4. **Press the toaster switch**: 토스터 스위치 누르기
5. **Open the oven**: 오븐 열기
6. **Pick up the eggplant**: 가지 집어 테이블에 놓기
7. **Pick up the potato**: 감자 집어 도마에 놓기

#### 성능 특성
- **배경 적응**: 새로운 배경에서의 안정적 작업 수행
- **일반화 능력**: 다양한 배경에서의 일관된 성능
- **시각적 적응**: 새로운 시각적 환경에 대한 적응 능력

### Fig. 18: 미확인 방해물과 객체에서의 질적 결과
> **인용**: "Fig. 18: Qualitative results for unseen distractors and objects." (논문 Fig. 18)

#### Fig. 18의 구성
- **7개 작업**: 미확인 방해물과 객체에서의 조작 작업
- **방해물 테스트**: 새로운 방해물이 있는 환경에서의 작업 수행
- **객체 다양성**: 다양한 새로운 객체에 대한 조작 능력

#### 주요 작업 분석
1. **Pick up the banana**: 바나나 집어 초록 접시에 놓기
2. **Pick up the tiger**: 호랑이 인형 집어 빨간 접시에 놓기
3. **Pick up the yellow bottle**: 노란 병 집어 테이블에 놓기
4. **Pick up the white bottle**: 흰 병 집어 야채 바구니에 놓기
5. **Open the drawer**: 서랍 열기
6. **Press the toaster switch**: 토스터 스위치 누르기
7. **Pick up the knife**: 칼 집어 서랍에 넣기

#### 성능 특성
- **방해물 적응**: 새로운 방해물이 있는 환경에서의 작업 수행
- **객체 다양성**: 다양한 새로운 객체에 대한 조작 능력
- **일반화 능력**: 미확인 환경에서의 안정적 성능

### Fig. 19: 전형적인 실패 사례
> **인용**: "Fig. 19: Our model exhibits several typical failure cases. For instance, it might prematurely close the gripper, fail to accurately grasp the target object, exhibit repeated oscillations, or successfully pick up an object but cannot place it in the correct location." (논문 Fig. 19)

#### Fig. 19의 구성
- **7개 실패 사례**: 다양한 실패 유형의 시각화
- **실패 유형**: 그리퍼 조기 닫힘, 물체 파악 실패, 반복적 진동, 잘못된 위치 배치
- **이중 시점**: Side Camera와 Wrist Camera의 실패 과정 시각화

#### 주요 실패 사례 분석
1. **칼 집어 서랍에 넣기 실패**: 그리퍼의 조기 닫힘으로 인한 파악 실패
2. **오븐 닫기 실패**: 오븐 문을 닫지 못하는 실패
3. **빨간 사과 집기**: 파악은 성공했으나 불안정한 파악
4. **야채 바구니에서 사과 집기**: 파악의 정확성 부족
5. **도마에서 감자 집기**: 파악의 안정성 부족
6. **흰 상자에서 초록 병 집기**: 파악의 정확성 부족
7. **양념 가루 붓기 실패**: 붓기 동작의 실패

#### 실패 유형 분석
- **그리퍼 조기 닫힘**: 목표 물체에 도달하기 전 그리퍼가 닫히는 현상
- **물체 파악 실패**: 목표 물체를 정확하게 잡지 못하는 현상
- **반복적 진동**: 그리퍼가 진동하며 안정적 파악에 실패하는 현상
- **잘못된 위치 배치**: 물체를 성공적으로 집었으나 올바른 위치에 놓지 못하는 현상

## 7. 실제 환경 실험의 종합 분석

### 성공 사례의 특성
1. **정확한 조작**: 각 작업의 순차적 실행 과정
2. **안정적 파지**: 다양한 물체의 안정적 파지
3. **정밀한 배치**: 목표 위치에의 정확한 배치
4. **일반화 능력**: 다양한 환경에서의 일관된 성능

### 실패 사례의 특성
1. **그리퍼 조기 닫힘**: 목표 물체에 도달하기 전 그리퍼가 닫히는 현상
2. **물체 파악 실패**: 목표 물체를 정확하게 잡지 못하는 현상
3. **반복적 진동**: 그리퍼가 진동하며 안정적 파악에 실패하는 현상
4. **잘못된 위치 배치**: 물체를 성공적으로 집었으나 올바른 위치에 놓지 못하는 현상

### 개선 방향
1. **그리퍼 제어**: 더 정밀한 그리퍼 제어 알고리즘 개발
2. **물체 인식**: 더 정확한 물체 인식 및 위치 추정
3. **안정성 향상**: 더 안정적인 파지 및 배치 알고리즘 개발
4. **일반화 능력**: 더 강건한 일반화 능력 개발

## 8. 결론

### Fig. 14-19의 핵심 의의
1. **시각적 검증**: 실제 환경에서의 다양한 실험 조건 시각화
2. **일반화 평가**: 로봇 정책의 일반화 능력 평가
3. **작업 다양성**: 20가지 다양한 조작 작업
4. **실패 분석**: 전형적인 실패 사례의 체계적 분석

### 연구의 의의
1. **실용성**: 실제 환경에서의 실용성 입증
2. **일반화**: 다양한 조건에서의 일반화 능력
3. **성능 검증**: 정량적 지표와 질적 결과의 일치
4. **실패 분석**: 실패 사례의 체계적 분석을 통한 개선 방향 제시

### 미래 연구 방향
1. **작업 확장**: 더 복잡한 조작 작업으로의 확장
2. **환경 다양화**: 더 다양한 환경에서의 성능 검증
3. **성능 향상**: 더 정밀하고 안정적인 조작 능력 개발
4. **실패 감소**: 실패 사례 분석을 통한 성능 개선

---

*분석 작성일: 2024년 12월*  
*원본 논문: "Towards Generalist Robot Policies: What Matters in Building Vision-Language-Action Models"*
