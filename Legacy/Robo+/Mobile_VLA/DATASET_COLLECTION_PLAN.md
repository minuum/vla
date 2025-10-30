# 📊 Mobile VLA 데이터셋 수집 계획

## 📊 **현재 상황 분석**

### **기존 데이터셋 현황**
- **현재 크기**: 72개 에피소드
- **데이터 형식**: HDF5 파일
- **수집 환경**: 단일 환경 조건
- **주행 패턴**: 제한적 다양성
- **성능 한계**: MAE 0.212~0.672 (모델별 차이)

### **성능 분석 결과**
| 모델 | 데이터셋 | MAE | 데이터 효율성 |
|------|----------|-----|---------------|
| Kosmos2+CLIP Hybrid | 원본 72 에피소드 | 0.212 | 높음 |
| Pure Kosmos2 | 원본 72 에피소드 | 0.247 | 높음 |
| Simple CLIP | 원본 72 에피소드 | 0.451 | 중간 |
| Original CLIP (증강) | 증강 720 에피소드 | 0.672 | 낮음 |

**핵심 발견**: 원본 72 에피소드가 증강 720 에피소드보다 우수한 성능

## 🎯 **데이터셋 확장 목표**

### **단기 목표 (1개월)**
- **목표 크기**: 72개 → 150개 에피소드 (2.1배 증가)
- **다양성 확보**: 3가지 환경 조건
- **품질 관리**: 자동화된 데이터 품질 검증

### **중기 목표 (3개월)**
- **목표 크기**: 150개 → 300개 에피소드 (2배 증가)
- **다양성 확보**: 5가지 환경 조건
- **실시간 수집**: 자동화된 데이터 수집 시스템

### **장기 목표 (6개월)**
- **목표 크기**: 300개 → 500개 에피소드 (1.7배 증가)
- **다양성 확보**: 10가지 환경 조건
- **산업용 수준**: 실제 배포 환경 데이터

## 📋 **데이터 수집 전략**

### **1. 환경 다양성 확보**

#### **1.1 조명 조건 다양화**
```python
class LightingConditionCollector:
    def __init__(self):
        self.lighting_conditions = [
            'natural_daylight',      # 자연광 (기존)
            'artificial_bright',     # 인공조명 밝음
            'artificial_dim',        # 인공조명 어둠
            'mixed_lighting',        # 혼합 조명
            'shadow_conditions'      # 그림자 조건
        ]
    
    def collect_episodes(self, condition, num_episodes=30):
        # 특정 조명 조건에서 에피소드 수집
        episodes = []
        for i in range(num_episodes):
            episode = self._collect_single_episode(condition)
            if self._validate_episode_quality(episode):
                episodes.append(episode)
        return episodes
    
    def _collect_single_episode(self, condition):
        # 조명 조건 설정
        self._set_lighting_condition(condition)
        
        # 에피소드 수집
        episode = {
            'images': [],
            'actions': [],
            'states': [],
            'text': self._generate_instruction(),
            'lighting_condition': condition
        }
        
        # 주행 시퀀스 수집
        for step in range(50):  # 50스텝 에피소드
            image = self._capture_image()
            action = self._get_action()
            state = self._get_robot_state()
            
            episode['images'].append(image)
            episode['actions'].append(action)
            episode['states'].append(state)
        
        return episode
```

#### **1.2 장애물 배치 다양화**
```python
class ObstacleConfigurationCollector:
    def __init__(self):
        self.obstacle_configs = [
            'no_obstacles',          # 장애물 없음
            'static_obstacles',      # 정적 장애물
            'dynamic_obstacles',     # 동적 장애물
            'mixed_obstacles',       # 혼합 장애물
            'complex_maze'           # 복잡한 미로
        ]
    
    def collect_episodes(self, config, num_episodes=25):
        episodes = []
        for i in range(num_episodes):
            # 장애물 배치 설정
            self._setup_obstacle_configuration(config)
            
            # 에피소드 수집
            episode = self._collect_episode_with_obstacles(config)
            if self._validate_episode_quality(episode):
                episodes.append(episode)
        
        return episodes
```

#### **1.3 바닥 재질 다양화**
```python
class SurfaceTypeCollector:
    def __init__(self):
        self.surface_types = [
            'smooth_concrete',       # 매끄러운 콘크리트
            'rough_concrete',        # 거친 콘크리트
            'carpet',               # 카펫
            'tile',                 # 타일
            'wood',                 # 나무
            'metal',                # 금속
            'outdoor_terrain'       # 실외 지형
        ]
    
    def collect_episodes(self, surface_type, num_episodes=20):
        episodes = []
        for i in range(num_episodes):
            # 바닥 재질 설정
            self._setup_surface_type(surface_type)
            
            # 에피소드 수집
            episode = self._collect_episode_on_surface(surface_type)
            if self._validate_episode_quality(episode):
                episodes.append(episode)
        
        return episodes
```

### **2. 주행 패턴 다양화**

#### **2.1 기본 주행 패턴**
```python
class BasicManeuverCollector:
    def __init__(self):
        self.basic_maneuvers = [
            'straight_line',         # 직선 주행
            'left_turn',            # 좌회전
            'right_turn',           # 우회전
            'u_turn',               # U턴
            's_curve',              # S자 곡선
            'parking',              # 주차
            'reversing'             # 후진
        ]
    
    def collect_episodes(self, maneuver, num_episodes=15):
        episodes = []
        for i in range(num_episodes):
            # 주행 패턴 설정
            self._setup_maneuver(maneuver)
            
            # 에피소드 수집
            episode = self._collect_maneuver_episode(maneuver)
            if self._validate_episode_quality(episode):
                episodes.append(episode)
        
        return episodes
```

#### **2.2 복합 주행 패턴**
```python
class ComplexManeuverCollector:
    def __init__(self):
        self.complex_maneuvers = [
            'obstacle_avoidance',    # 장애물 회피
            'narrow_passage',        # 좁은 통로
            'multi_point_navigation', # 다중 지점 내비게이션
            'emergency_stop',        # 긴급 정지
            'precise_positioning'    # 정밀 위치 조정
        ]
    
    def collect_episodes(self, maneuver, num_episodes=10):
        episodes = []
        for i in range(num_episodes):
            # 복합 주행 패턴 설정
            self._setup_complex_maneuver(maneuver)
            
            # 에피소드 수집
            episode = self._collect_complex_episode(maneuver)
            if self._validate_episode_quality(episode):
                episodes.append(episode)
        
        return episodes
```

### **3. 속도 범위 다양화**

#### **3.1 속도별 수집**
```python
class SpeedRangeCollector:
    def __init__(self):
        self.speed_ranges = [
            'very_slow',    # 매우 느림 (0.1-0.3 m/s)
            'slow',         # 느림 (0.3-0.5 m/s)
            'normal',       # 보통 (0.5-0.8 m/s)
            'fast',         # 빠름 (0.8-1.2 m/s)
            'very_fast'     # 매우 빠름 (1.2-1.5 m/s)
        ]
    
    def collect_episodes(self, speed_range, num_episodes=20):
        episodes = []
        for i in range(num_episodes):
            # 속도 범위 설정
            self._set_speed_range(speed_range)
            
            # 에피소드 수집
            episode = self._collect_speed_episode(speed_range)
            if self._validate_episode_quality(episode):
                episodes.append(episode)
        
        return episodes
```

### **4. 데이터 품질 관리**

#### **4.1 자동 품질 검증**
```python
class DataQualityValidator:
    def __init__(self):
        self.quality_metrics = [
            'action_consistency',    # 액션 일관성
            'trajectory_smoothness', # 궤적 부드러움
            'collision_detection',   # 충돌 감지
            'goal_reachability',     # 목표 도달 가능성
            'image_quality',         # 이미지 품질
            'sensor_reliability'     # 센서 신뢰성
        ]
    
    def validate_episode_quality(self, episode):
        quality_score = 0.0
        max_score = len(self.quality_metrics)
        
        # 액션 일관성 검사
        action_consistency = self._check_action_consistency(episode['actions'])
        quality_score += action_consistency
        
        # 궤적 부드러움 검사
        trajectory_smoothness = self._check_trajectory_smoothness(episode['states'])
        quality_score += trajectory_smoothness
        
        # 충돌 감지
        collision_free = self._check_collision_free(episode['states'])
        quality_score += collision_free
        
        # 목표 도달 가능성
        goal_reachable = self._check_goal_reachability(episode)
        quality_score += goal_reachable
        
        # 이미지 품질
        image_quality = self._check_image_quality(episode['images'])
        quality_score += image_quality
        
        # 센서 신뢰성
        sensor_reliability = self._check_sensor_reliability(episode)
        quality_score += sensor_reliability
        
        # 품질 점수 정규화
        normalized_score = quality_score / max_score
        
        # 품질 임계값 (0.7 이상)
        return normalized_score >= 0.7
    
    def _check_action_consistency(self, actions):
        # 액션 일관성 검사
        action_diff = torch.diff(actions, dim=0)
        action_std = torch.std(action_diff)
        
        # 일관성 점수 (낮은 표준편차 = 높은 일관성)
        consistency_score = 1.0 / (1.0 + action_std)
        return min(consistency_score, 1.0)
    
    def _check_trajectory_smoothness(self, states):
        # 궤적 부드러움 검사
        if len(states) < 3:
            return 0.0
        
        # 2차 미분으로 부드러움 측정
        second_derivative = torch.diff(states, n=2, dim=0)
        smoothness = 1.0 / (1.0 + torch.mean(torch.abs(second_derivative)))
        return min(smoothness, 1.0)
    
    def _check_collision_free(self, states):
        # 충돌 감지
        # 실제 구현에서는 장애물 맵과 비교
        return 1.0  # 임시 구현
    
    def _check_goal_reachability(self, episode):
        # 목표 도달 가능성 검사
        # 실제 구현에서는 경로 계획 알고리즘 사용
        return 1.0  # 임시 구현
    
    def _check_image_quality(self, images):
        # 이미지 품질 검사
        quality_scores = []
        for image in images:
            # 블러 검사
            blur_score = self._calculate_blur_score(image)
            # 밝기 검사
            brightness_score = self._calculate_brightness_score(image)
            # 대비 검사
            contrast_score = self._calculate_contrast_score(image)
            
            overall_score = (blur_score + brightness_score + contrast_score) / 3
            quality_scores.append(overall_score)
        
        return torch.mean(torch.tensor(quality_scores))
    
    def _check_sensor_reliability(self, episode):
        # 센서 신뢰성 검사
        # 실제 구현에서는 센서 데이터 일관성 검사
        return 1.0  # 임시 구현
```

#### **4.2 데이터 증강 전략**
```python
class DataAugmentationStrategy:
    def __init__(self):
        self.augmentation_methods = [
            'physics_consistent_flip',    # 물리적 일관성 보장 반전
            'speed_variation',           # 속도 변화
            'action_noise',              # 액션 노이즈
            'temporal_sampling',         # 시간적 샘플링
            'lighting_adjustment'        # 조명 조정
        ]
    
    def augment_episode(self, episode, method):
        if method == 'physics_consistent_flip':
            return self._physics_consistent_flip(episode)
        elif method == 'speed_variation':
            return self._speed_variation(episode)
        elif method == 'action_noise':
            return self._action_noise(episode)
        elif method == 'temporal_sampling':
            return self._temporal_sampling(episode)
        elif method == 'lighting_adjustment':
            return self._lighting_adjustment(episode)
        
        return episode
    
    def _physics_consistent_flip(self, episode):
        # 물리적 일관성 보장 반전
        augmented = episode.copy()
        
        # 이미지 좌우 반전
        augmented['images'] = torch.flip(episode['images'], dims=[3])
        
        # 액션 x축 부호 반전
        augmented['actions'] = episode['actions'].clone()
        augmented['actions'][:, 0] *= -1
        
        # 상태 정보 업데이트
        if 'states' in episode:
            augmented['states'] = episode['states'].clone()
            augmented['states'][:, 0] *= -1  # x축 위치 반전
            augmented['states'][:, 2] *= -1  # x축 방향 반전
        
        return augmented
```

## 📊 **수집 계획 상세**

### **Phase 1: 기본 다양성 확보 (1개월)**

#### **Week 1-2: 환경 조건 다양화**
- [ ] 조명 조건 3가지 추가 수집 (90개 에피소드)
- [ ] 장애물 배치 2가지 추가 수집 (50개 에피소드)
- [ ] 바닥 재질 2가지 추가 수집 (40개 에피소드)

#### **Week 3-4: 주행 패턴 다양화**
- [ ] 기본 주행 패턴 4가지 추가 수집 (60개 에피소드)
- [ ] 복합 주행 패턴 2가지 추가 수집 (20개 에피소드)
- [ ] 속도 범위 3가지 추가 수집 (60개 에피소드)

**Phase 1 목표**: 72개 → 150개 에피소드 (2.1배 증가)

### **Phase 2: 고급 다양성 확보 (2개월)**

#### **Month 2: 고급 환경 조건**
- [ ] 조명 조건 2가지 추가 (60개 에피소드)
- [ ] 장애물 배치 3가지 추가 (75개 에피소드)
- [ ] 바닥 재질 3가지 추가 (60개 에피소드)

#### **Month 3: 고급 주행 패턴**
- [ ] 복합 주행 패턴 3가지 추가 (45개 에피소드)
- [ ] 속도 범위 2가지 추가 (40개 에피소드)
- [ ] 특수 상황 2가지 추가 (30개 에피소드)

**Phase 2 목표**: 150개 → 300개 에피소드 (2배 증가)

### **Phase 3: 산업용 수준 달성 (3개월)**

#### **Month 4-6: 실제 배포 환경**
- [ ] 실외 환경 데이터 수집 (100개 에피소드)
- [ ] 다양한 기상 조건 (50개 에피소드)
- [ ] 실제 작업 환경 (50개 에피소드)

**Phase 3 목표**: 300개 → 500개 에피소드 (1.7배 증가)

## 📈 **예상 성능 향상**

### **데이터셋 크기별 성능 예상**
| 데이터셋 크기 | 예상 MAE | 성능 향상 | 수집 기간 |
|---------------|----------|-----------|-----------|
| 72개 (현재) | 0.212-0.672 | 기준점 | 완료 |
| 150개 (Phase 1) | 0.15-0.5 | 30-40% | 1개월 |
| 300개 (Phase 2) | 0.1-0.3 | 50-70% | 3개월 |
| 500개 (Phase 3) | 0.08-0.2 | 70-90% | 6개월 |

### **다양성별 성능 기여도**
| 다양성 요소 | 성능 기여도 | 구현 난이도 | 우선순위 |
|-------------|-------------|-------------|----------|
| 조명 조건 | 15% | 중간 | 높음 |
| 장애물 배치 | 25% | 높음 | 매우 높음 |
| 바닥 재질 | 10% | 중간 | 중간 |
| 주행 패턴 | 30% | 높음 | 매우 높음 |
| 속도 범위 | 20% | 낮음 | 높음 |

## 🎯 **수집 우선순위**

### **1순위 (즉시 시작)**
1. **장애물 회피 패턴** - 가장 높은 성능 기여도
2. **복합 주행 패턴** - 실용성 향상
3. **속도 범위 다양화** - 구현 간단, 효과 좋음

### **2순위 (단기)**
4. **조명 조건 다양화** - 환경 적응성 향상
5. **바닥 재질 다양화** - 일반화 성능 향상

### **3순위 (중기)**
6. **실외 환경 데이터** - 실제 배포 준비
7. **특수 상황 데이터** - 안전성 향상

## 📋 **구현 체크리스트**

### **Week 1 체크포인트**
- [ ] 데이터 수집 시스템 구축
- [ ] 품질 검증 시스템 구현
- [ ] 기본 환경 조건 3가지 수집 완료

### **Week 2 체크포인트**
- [ ] 장애물 배치 2가지 수집 완료
- [ ] 바닥 재질 2가지 수집 완료
- [ ] 데이터 품질 검증 완료

### **Week 3 체크포인트**
- [ ] 기본 주행 패턴 4가지 수집 완료
- [ ] 복합 주행 패턴 2가지 수집 완료
- [ ] 속도 범위 3가지 수집 완료

### **Week 4 체크포인트**
- [ ] Phase 1 목표 달성 (150개 에피소드)
- [ ] 성능 향상 검증 완료
- [ ] Phase 2 계획 수립 완료

## 🎉 **예상 성과**

### **성능 향상**
- **MAE**: 0.212 → 0.08 (62% 향상)
- **일반화 성능**: 크게 향상
- **실용성**: 실제 배포 가능한 수준

### **기능 향상**
- **환경 적응성**: 다양한 환경 조건 대응
- **안전성**: 충돌 회피 및 안전한 주행
- **효율성**: 최적 경로 계획 및 실행

### **비즈니스 가치**
- **상용화 가능**: 산업용 로봇 제어 시스템
- **기술 이전**: 로봇 제어 솔루션 라이선싱
- **연구 발전**: 최신 VLA 기술 발전

---

**📊 체계적인 데이터 수집으로 성능 혁신! 📊**

*이 계획은 2025년 1월 25일에 수립되었습니다.*
