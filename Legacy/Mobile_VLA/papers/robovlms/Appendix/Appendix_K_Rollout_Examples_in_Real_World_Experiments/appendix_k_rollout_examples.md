# Appendix K: 실제 환경 실험의 롤아웃 예시

> 인용: 논문 Appendix K - ROLLOUT EXAMPLES IN REAL-WORLD EXPERIMENTS

## **실험 개요**

> **인용**: "Fig. 15: This figure illustrates the experimental setup of some real-world tasks. The models are evaluated across 20 tasks, each with 5 rollouts, involving unseen distractors, unseen backgrounds, unseen target objects, and novel skill descriptions. Note that some tasks exclude the unseen target object setting due to the lack of suitable alternative unseen objects." (Appendix K)

- **총 작업**: 20개 작업
- **롤아웃**: 각 작업당 5회 롤아웃
- **평가 조건**: 4가지 Unseen 설정 + 1가지 Basic 설정
- **제외 사항**: 일부 작업은 적절한 대안 객체 부족으로 unseen target object 설정 제외

## **실험 설정 구조**

### **5가지 실험 조건**

1. **Basic**: 표준, 깨끗한 환경에서 지정된 객체로 작업 수행
2. **Unseen Distractor**: 관련 없는 추가 객체를 환경에 도입하여 로봇의 집중력 테스트
3. **Unseen Background**: 환경의 시각적 배경 변화에 대한 로봇의 강건성 평가
4. **Unseen Target Object**: 학습된 스킬을 이전에 보지 못한 새로운 객체에 적용
5. **Novel Skill Description**: 의미적으로 유사하지만 새로운 언어 지시사항으로 작업 이해 및 실행 테스트

## **20개 작업 상세 분석**

### **객체 집기 작업 (Object Picking)**

#### **1. Pick up the knife**
- **Basic**: 나무 도마 위의 칼을 집기
- **Unseen Distractor**: 사과와 녹색 병이 방해물로 추가
- **Unseen Background**: 파란색과 흰색 패턴 천 배경
- **Unseen Target Object**: "Pick up the yellow bottle" (칼 대신 노란 병)
- **Novel Skill Description**: "Take the knife" (Pick up → Take)

#### **2. Pick up the cucumber**
- **Basic**: 오이 집기
- **Unseen Distractor**: 방해물 추가
- **Unseen Background**: 배경 변화
- **Unseen Target Object**: "Pick up the white bottle" (오이 대신 흰 병)
- **Novel Skill Description**: "Lift the cucumber" (Pick up → Lift)

#### **3. Pick up the eggplant**
- **Basic**: 가지 집기
- **Unseen Distractor**: 방해물 추가
- **Unseen Background**: 배경 변화
- **Unseen Target Object**: "Pick up the white bottle" (가지 대신 흰 병)
- **Novel Skill Description**: "Lift the eggplant" (Pick up → Lift)

#### **4. Pick up the green bottle**
- **Basic**: 녹색 병 집기
- **Unseen Distractor**: 방해물 추가
- **Unseen Background**: 배경 변화
- **Unseen Target Object**: 새로운 객체로 스킬 적용
- **Novel Skill Description**: 새로운 언어 표현

#### **5. Pick up the red cup**
- **Basic**: 빨간 컵 집기
- **Unseen Distractor**: 방해물 추가
- **Unseen Background**: 배경 변화
- **Unseen Target Object**: 새로운 객체로 스킬 적용
- **Novel Skill Description**: 새로운 언어 표현

#### **6. Pick up the green cup**
- **Basic**: 녹색 컵 집기
- **Unseen Distractor**: 방해물 추가
- **Unseen Background**: 배경 변화
- **Unseen Target Object**: 새로운 객체로 스킬 적용
- **Novel Skill Description**: 새로운 언어 표현

#### **7. Pick up the mandarin**
- **Basic**: 귤 집기
- **Unseen Distractor**: 방해물 추가
- **Unseen Background**: 배경 변화
- **Unseen Target Object**: 새로운 객체로 스킬 적용
- **Novel Skill Description**: 새로운 언어 표현

#### **8. Pick up the potato**
- **Basic**: 감자 집기
- **Unseen Distractor**: 방해물 추가
- **Unseen Background**: 배경 변화
- **Unseen Target Object**: 새로운 객체로 스킬 적용
- **Novel Skill Description**: 새로운 언어 표현

#### **9. Pick up the seasoning powder**
- **Basic**: 조미료 가루 집기
- **Unseen Distractor**: 방해물 추가
- **Unseen Background**: 배경 변화
- **Unseen Target Object**: 새로운 객체로 스킬 적용
- **Novel Skill Description**: 새로운 언어 표현

### **관절 객체/스위치 상호작용 작업**

#### **10. Press the toaster switch**
- **Basic**: 토스터 스위치 누르기
- **Unseen Distractor**: 방해물 추가
- **Unseen Background**: 배경 변화
- **Unseen Target Object**: 새로운 객체로 스킬 적용
- **Novel Skill Description**: 새로운 언어 표현

#### **11. Open the oven**
- **Basic**: 오븐 열기
- **Unseen Distractor**: 방해물 추가
- **Unseen Background**: 배경 변화
- **Unseen Target Object**: 새로운 객체로 스킬 적용
- **Novel Skill Description**: 새로운 언어 표현

#### **12. Shut the oven**
- **Basic**: 오븐 닫기
- **Unseen Distractor**: 방해물 추가
- **Unseen Background**: 배경 변화
- **Unseen Target Object**: 새로운 객체로 스킬 적용
- **Novel Skill Description**: 새로운 언어 표현

#### **13. Open the drawer**
- **Basic**: 서랍 열기
- **Unseen Distractor**: 방해물 추가
- **Unseen Background**: 배경 변화
- **Unseen Target Object**: 새로운 객체로 스킬 적용
- **Novel Skill Description**: "Unlock the drawer" (Open → Unlock)

### **붓기/뿌리기 작업**

#### **14. Pour the seasoning powder**
- **Basic**: 조미료 가루 붓기
- **Unseen Distractor**: 방해물 추가
- **Unseen Background**: 배경 변화
- **Unseen Target Object**: 새로운 객체로 스킬 적용
- **Novel Skill Description**: "Sprinkle the seasoning powder" (Pour → Sprinkle)

## **시각적 요소 및 세부사항**

### **로봇 시스템**
- **로봇 팔**: 검은색 로봇 팔
- **그리퍼**: 2개 손가락 그리퍼
- **일관성**: 모든 작업에서 일관되게 보임

### **객체 종류**
- **일상용품**: 칼, 오이, 가지, 병, 컵, 귤, 감자, 조미료 가루
- **가전제품**: 토스터, 오븐
- **가구**: 서랍이 있는 작은 캐비닛

### **환경 설정**
- **테이블탑**: 대부분의 작업이 테이블탑에서 수행
- **특수 환경**: 토스터, 오븐, 서랍이 있는 캐비닛
- **바운딩 박스**: 노란색 바운딩 박스로 대상 객체 또는 상호작용 영역 강조

## **평가 방법론**

### **롤아웃 구조**
- **총 롤아웃**: 20개 작업 × 5회 = 100회 롤아웃
- **각 롤아웃**: 5가지 실험 조건 (Basic + 4가지 Unseen)
- **총 실험**: 100 × 5 = 500회 실험

### **성능 지표**
- **성공률**: 각 설정에 대한 평균 성공률
- **일반화 능력**: Unseen 조건에서의 성능
- **강건성**: 다양한 환경 변화에 대한 적응력

## **실험의 의의**

### **일반화 능력 평가**
- **Unseen Distractors**: 주의 집중력과 선택적 인식 능력
- **Unseen Backgrounds**: 시각적 변화에 대한 강건성
- **Unseen Target Objects**: 학습된 스킬의 새로운 객체 적용 능력
- **Novel Skill Descriptions**: 언어 이해의 유연성

### **실용적 가치**
- **실제 환경**: 시뮬레이션이 아닌 실제 로봇 환경에서의 평가
- **다양성**: 20가지 서로 다른 조작 작업
- **도전성**: 4가지 Unseen 조건으로 어려운 일반화 시나리오
- **포괄성**: 객체 조작, 관절 객체 상호작용, 붓기 등 다양한 스킬

## **제한사항 및 고려사항**

### **Unseen Target Object 제외**
> **인용**: "Note that some tasks exclude the unseen target object setting due to the lack of suitable alternative unseen objects." (Appendix K)

- **제외 작업**: Open Drawer와 같은 작업
- **이유**: 적절한 대안 객체 부족
- **적용 범위**: 주로 picking 작업에만 적용

### **객체 레이아웃**
- **랜덤 초기화**: 각 롤아웃의 객체 레이아웃이 훈련 세트와 다르게 랜덤 초기화
- **일관성**: 실험의 공정성을 위한 표준화된 설정
