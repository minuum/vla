# Mobile-VLA 데이터 증강 전략

## 📊 현재 상태 및 목표

### 현재 데이터셋
- **수량**: 약 500개 에피소드 (175 train, 44 validation 실제 사용)
- **수집 방법**: 실제 로봇 주행 데이터 (H5 형식)
- **문제점**: 
  - 데이터 부족으로 인한 일반화 성능 제한
  - 다양한 환경/상황에 대한 노출 부족
  - 수집 비용 및 시간 문제

### 증강 목표
- **목표 데이터셋 크기**: 500개 → **5,000개** (10배 증강)
- **증강 방법**: 시뮬레이션 기반 합성 데이터 생성 + 이미지 증강
- **기대 효과**: 
  - 모델 일반화 성능 향상
  - 다양한 환경/장애물에 대한 강건성 확보
  - Sim-to-Real 전이 학습 검증

---

## 🔬 VLA 데이터 증강 실제 사례

### 1. OpenVLA - MimicGen 활용
**출처**: [OpenVLA Project](https://openvla.github.io)

#### 방법론
- **MimicGen 시뮬레이터**: Robomimic + Robosuite 기반
- **동작 원리**:
  1. 전문가 시연(demonstration)을 객체 중심 서브태스크로 분해
  2. 서브태스크를 새로운 장면에 변환(transform) 및 재조합(recompose)
  3. 수천 개의 고유한 궤적 자동 합성

#### Domain Randomization 적용
- **색상/재질 변화**: 물체의 색상과 재질을 무작위로 변경하여 외형이 아닌 속성 학습 유도
- **자연어 레이블 변형**: 동일한 동작에 대해 다양한 문장 표현 생성
- **적용 태스크**: Stack_D2, Stack_D3, Stack_D4 (쌓기 작업)

#### 검증 방법
- **Agent Studio**: 온라인 시뮬레이션 환경에서 성공률 측정
- **보상 신호 기반**: 에피소드 성공 여부를 자동으로 카운트

---

### 2. Google ROSIE - Diffusion Model 기반 증강
**출처**: [ROSIE: Robot Learning with Semantically Imagined Experience](https://roboticsproceedings.org)

#### 방법론
- **생성 모델**: DALL-E 2, Imagen, StableDiffusion 등 텍스트-이미지 diffusion 모델
- **동작 원리**:
  1. 기존 로봇 데이터셋에서 변경할 영역 식별
  2. 사람이 제공한 지시사항(instruction) 기반 inpainting
  3. 새로운 태스크, 방해 요소(distractors), 배경 생성

#### 증강 효과
- ✅ 새로운 물체에 대한 일반화 능력 향상
- ✅ 방해 요소에 대한 강건성 증가
- ✅ 다양한 배경에서의 조작 성능 개선

#### 주요 장점
- **Photorealistic**: 실사와 유사한 품질의 이미지 생성
- **Semantic Control**: 의미 있는 변형만 적용 가능
- **Real Data 기반**: 실제 데이터에서 출발하여 Sim-to-Real 갭 최소화

---

### 3. IA-VLA - Input Augmentation Framework
**출처**: [IA-VLA: Input Augmentation for Vision-Language-Action Models](https://arxiv.org/abs/...)

#### 방법론
- **전처리 단계**: 더 큰 Vision-Language Model을 전처리기로 활용
- **Semantic Segmentation**: Semantic-SAM으로 객체 인스턴스 분할
- **Numeric Tagging**: 각 객체 마스크에 고유 번호 태그 부여
- **Mask Filtering**: 
  - Mask Patch Filter: 단편화된 객체 분리
  - Mask Overlap Filter: 중복 마스크 해결

#### 적용 사례
- 시각적으로 유사한 물체가 많은 태스크
- 공간적 언어 명령이 복잡한 태스크 (예: "왼쪽 두 번째 블록")

---

### 4. Cross-Embodiment 데이터 활용
**출처**: RT-X Dataset

#### 개념
- **RT-X 데이터셋**: 22개 로봇 플랫폼의 100만+ 궤적 수집
- **통합 액션 공간**: 서로 다른 로봇의 액션 표현을 표준화
- **전이 학습**: 하나의 VLA 모델로 다양한 로봇 제어 가능

#### Mobile-VLA 적용 가능성
- 다른 모바일 로봇 데이터셋과의 통합 가능성
- 2DOF 액션 공간의 표준화/정규화 전략 참고

---

## 🚀 Mobile-VLA 증강 전략

### Phase 1: 시뮬레이션 환경 구축 (3-4주)

#### 추천 시뮬레이터
1. **Isaac Gym / Isaac Lab (NVIDIA)**
   - ✅ GPU 병렬화로 빠른 데이터 생성
   - ✅ Photorealistic 렌더링 지원 (NVIDIA Omniverse 연동)
   - ✅ 모바일 로봇 내비게이션에 최적화
   - ⚠️ 초기 설정 복잡도 높음

2. **Habitat-AI (Meta)**
   - ✅ 실내 내비게이션 전문 플랫폼
   - ✅ 다양한 3D 환경 씬 제공 (Matterport3D, Gibson)
   - ✅ 실제 건물 스캔 데이터 활용 가능
   - ✅ Python API 직관적

3. **PyBullet (경량 대안)**
   - ✅ 설치 및 사용 간편
   - ✅ 커스터마이징 유연성 높음
   - ⚠️ 렌더링 품질 낮음
   - ⚠️ 대규모 병렬화 제한적

#### 구현 작업
```python
# 예상 디렉토리 구조
simulation/
├── environments/           # 시뮬레이션 환경 정의
│   ├── office_env.py      # 사무실 환경
│   ├── hallway_env.py     # 복도 환경
│   └── outdoor_env.py     # 실외 환경
├── data_generator.py       # 데이터 생성 스크립트
├── domain_randomization.py # DR 파라미터 설정
└── configs/
    └── augmentation_config.yaml
```

---

### Phase 2: Domain Randomization 적용 (2주)

#### 시각적 변형 (Visual Randomization)
```yaml
visual_randomization:
  lighting:
    brightness: [0.5, 1.5]      # 조명 밝기
    shadow_intensity: [0.0, 1.0] # 그림자 강도
  
  camera:
    position_noise: 0.05         # 카메라 위치 노이즈 (m)
    rotation_noise: 5.0          # 카메라 각도 노이즈 (도)
    fov: [60, 90]                # 시야각 변화
  
  textures:
    floor_materials: ["wood", "tile", "carpet", "concrete"]
    wall_colors: ["#FFFFFF", "#F0F0F0", "#E0E0E0"]
  
  weather_effects:              # 실외용
    rain_probability: 0.1
    fog_density: [0.0, 0.3]
```

#### 물리적 변형 (Physics Randomization)
```yaml
physics_randomization:
  robot:
    mass: [10.0, 15.0]           # 로봇 질량 (kg)
    friction: [0.5, 1.5]         # 바퀴 마찰 계수
    max_velocity: [0.8, 1.2]     # 최대 속도 (m/s)
  
  obstacles:
    static_count: [5, 20]        # 고정 장애물 수
    dynamic_count: [0, 5]        # 동적 장애물 수
    size_range: [0.1, 0.5]       # 장애물 크기 (m)
```

#### 시나리오 변형 (Scenario Randomization)
- **목표 위치**: 에피소드마다 무작위 목표 지점 설정
- **시작 위치**: 다양한 초기 위치/각도에서 출발
- **경로 복잡도**: 장애물 배치 패턴 변경

---

### Phase 3: Diffusion Model 기반 이미지 증강 (3주)

#### ControlNet + StableDiffusion 활용
**목표**: 실제 데이터의 배경/조명/물체를 변형하여 다양성 확보

#### 구현 프로세스
1. **Depth Map 추출**
   ```python
   # 실제 이미지에서 깊이 맵 생성
   from transformers import pipeline
   
   depth_estimator = pipeline("depth-estimation", 
                              model="Intel/dpt-large")
   depth_map = depth_estimator(original_image)
   ```

2. **ControlNet으로 이미지 변형**
   ```python
   # 깊이 맵 유지하며 배경/조명 변경
   from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
   
   controlnet = ControlNetModel.from_pretrained(
       "lllyasviel/control_v11f1p_sd15_depth"
   )
   
   pipe = StableDiffusionControlNetPipeline.from_pretrained(
       "runwayml/stable-diffusion-v1-5",
       controlnet=controlnet
   )
   
   prompts = [
       "office hallway, bright fluorescent lighting",
       "dark corridor, evening light",
       "modern building interior, glass walls"
   ]
   
   for prompt in prompts:
       augmented_image = pipe(
           prompt=prompt,
           image=depth_map,
           num_inference_steps=20
       ).images[0]
   ```

3. **Action Labels 보존**
   - 이미지만 변형하고 원본 액션 레이블 유지
   - 깊이/구조 정보 보존으로 액션 유효성 보장

#### 증강 시나리오
- ☀️ 낮 → 밤 환경 전환
- 🏢 실내 → 실외 스타일 변경
- 🌦️ 날씨 효과 추가 (비, 안개)
- 👥 사람/동적 요소 추가

---

### Phase 4: 증강 데이터 검증 및 필터링 (1주)

#### 품질 검증 메트릭
```python
# 증강 데이터 품질 체크리스트
quality_checks = {
    "action_validity": check_action_bounds,      # 액션 범위 검증
    "trajectory_smoothness": check_smoothness,   # 궤적 연속성
    "collision_free": check_collision,           # 충돌 없음
    "goal_reachability": check_reachability,     # 목표 도달 가능성
    "visual_quality": check_image_quality        # 이미지 품질 (blur, noise)
}
```

#### 필터링 기준
- ❌ 물리적으로 불가능한 궤적 제거
- ❌ 과도하게 왜곡된 이미지 제거
- ❌ 목표 도달 실패 에피소드 제거
- ✅ 다양성 점수 기반 샘플링

---

## 📈 증강 효과 측정 계획

### Baseline (증강 전)
- LoRA 모델 (500 episodes)
- Validation Loss: 0.213
- Training Loss: 0.134

### Augmented Model (증강 후)
**학습 설정**:
- 500 real + 4,500 synthetic = 5,000 episodes
- LoRA 파인튜닝 유지
- Validation set은 100% 실제 데이터 유지

**측정 지표**:
1. **Validation Loss**: 감소 기대 (< 0.20)
2. **Generalization**: 새로운 환경에서의 성공률
3. **Robustness**: 장애물/조명 변화에 대한 강건성
4. **Sim-to-Real Gap**: 시뮬레이션 vs 실제 성능 차이

### A/B 테스트
| 모델 버전 | 데이터셋 | Val Loss | 새 환경 성공률 | 장애물 회피율 |
|----------|---------|----------|--------------|-------------|
| Baseline | 500 real | 0.213 | TBD | TBD |
| Aug v1 | +2000 sim (DR only) | TBD | TBD | TBD |
| Aug v2 | +2000 sim + 2500 diffusion | TBD | TBD | TBD |
| Final | Best combination | TBD | TBD | TBD |

---

## 🛠️ 구현 타임라인

### Week 1-2: 환경 구축
- [ ] Habitat-AI 또는 Isaac Sim 설치 및 설정
- [ ] 실제 환경과 유사한 맵 제작 (사무실/복도)
- [ ] 로봇 모델 import 및 센서 설정

### Week 3-4: Domain Randomization
- [ ] DR 파라미터 YAML 설정 파일 작성
- [ ] 자동 데이터 생성 파이프라인 구축
- [ ] 첫 1,000개 시뮬레이션 에피소드 생성

### Week 5-6: Diffusion 증강
- [ ] ControlNet 파이프라인 구축
- [ ] 실제 데이터 500개에 대해 다양한 스타일 적용
- [ ] 증강 이미지 품질 검증 및 필터링

### Week 7: 통합 및 학습
- [ ] 증강 데이터셋 통합 (H5 형식 변환)
- [ ] LoRA 모델 재학습
- [ ] 성능 비교 및 분석

### Week 8: 검증 및 보고서
- [ ] 실제 로봇 테스트 (가능 시)
- [ ] 증강 효과 분석 리포트 작성
- [ ] 논문용 데이터 정리

---

## 📚 참고 자료

### 논문
1. **OpenVLA**: "Open-Source Vision-Language-Action Model" (2024)
2. **ROSIE**: "Robot Learning with Semantically Imagined Experience" (RSS 2024)
3. **MimicGen**: "A Data Generation System for Scalable Robot Learning using Human Demonstrations" (CoRL 2023)
4. **IA-VLA**: "Input Augmentation for Vision-Language-Action Models" (2024)

### 시뮬레이터 문서
- [NVIDIA Isaac Gym](https://developer.nvidia.com/isaac-gym)
- [Meta Habitat-AI](https://aihabitat.org/)
- [PyBullet Quickstart](https://pybullet.org/wordpress/)

### Diffusion Models
- [ControlNet](https://github.com/lllyasviel/ControlNet)
- [StableDiffusion](https://github.com/Stability-AI/stablediffusion)

---

## ⚠️ 주의사항 및 한계

### Sim-to-Real Gap
- 시뮬레이션 데이터가 실제와 완벽히 일치하지 않음
- 센서 노이즈, 바퀴 슬립 등 실제 물리 현상 재현 한계
- **대책**: Domain Randomization을 충분히 적용하여 일반화 능력 향상

### 컴퓨팅 리소스
- 대규모 시뮬레이션 데이터 생성에는 GPU 필수
- Diffusion 모델 추론 시간 고려 (이미지당 5-10초)
- **대책**: 클라우드 GPU(Colab Pro, Lambda Labs) 활용

### 데이터 품질
- 자동 생성 데이터의 품질이 실제 데이터보다 낮을 수 있음
- 과도한 증강은 오히려 성능 저하 가능
- **대책**: 단계적 증강 + A/B 테스트로 최적 비율 탐색

---

**작성일**: 2025-11-25  
**작성자**: Mobile-VLA Research Team  
**다음 업데이트**: 환경 구축 완료 후 실험 결과 반영
