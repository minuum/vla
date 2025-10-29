# Mobile VLA 데이터 수집 현황 및 계획

## 1. 현재 상황

### 1.1 데이터 수집 환경
- **브랜치**: `b131fb5: fix : camera_service_server & vla_collector`
- **환경**: 후생관 301호실
- **상태**: 환경 구축 완료, 수집 준비 단계

### 1.2 수집 현황
- **목표**: 1000개 에피소드
- **현재**: 1000개 미달성
- **다음 단계**: 밤 시간대부터 수집 시작

## 2. 데이터 수집 계획

### 2.1 시간대별 수집 계획 (조정됨)

**밤 시간대 (20:00~22:00) - 우선순위 1**
- **조명**: 후생관 301호실 형광등
- **특징**: 태양광 없음, 일정한 조명 조건
- **목표**: 400개 에피소드 수집 (증가)
- **시작 시점**: 즉시 시작 가능

**새벽 시간대 (00:00~04:00) - 우선순위 2**
- **조명**: 형광등만
- **특징**: 가장 일정한 조명 조건
- **목표**: 400개 에피소드 수집 (증가)
- **시작 시점**: 밤 데이터 수집 후

**낮 시간대 (13:00~15:00) - 우선순위 3 (최소화)**
- **조명**: 형광등 + 자연광 혼합 (창 앞 테이핑 맵)
- **특징**: 조명 변화, 그림자 효과 (복잡성 증가)
- **목표**: 200개 에피소드 수집 (대폭 감소)
- **시작 시점**: 밤/새벽 데이터 수집 후

### 2.2 환경 설정

**물리적 환경**:
- **장애물**: 육면체 형태의 박스 2개
- **목표물**: 1.5L 제로콜라
- **공간**: 후생관 301호실
- **로봇**: Mobile Robot (2D 이동)

**카메라 설정**:
- **정적 카메라**: 천장 고정 (전체 작업 공간 시야)
- **그리퍼 카메라**: 로봇 팔 끝단 (선택적)
- **해상도**: 224x224 (CLIP 표준)
- **프레임레이트**: 30 FPS

## 3. 데이터 수집 프로세스

### 3.1 에피소드 구성

**에피소드 길이**: 50-100 스텝
**태스크 분류**:
```python
TASK_CATEGORIES = {
    "navigation": {
        "count": 400,           # 40%
        "tasks": [
            "go to coke bottle",
            "navigate around boxes",
            "go to corner",
            "return to start"
        ]
    },
    "manipulation": {
        "count": 300,           # 30%
        "tasks": [
            "pick up coke bottle",
            "place coke bottle",
            "grasp and move",
            "release object"
        ]
    },
    "combined": {
        "count": 300,           # 30%
        "tasks": [
            "go to coke and pick up",
            "navigate to box and place",
            "go around boxes and pick up",
            "complex navigation + manipulation"
        ]
    }
}
```

### 3.2 데이터 수집 스크립트

**환경 테스트**:
```bash
# Branch b131fb5로 전환
git checkout b131fb5

# 카메라 서비스 서버 실행
ros2 launch camera_service_server camera_service.launch.py

# VLA 컬렉터 실행
ros2 launch vla_collector vla_collector.launch.py
```

**데이터 수집 명령어**:
```python
# 에피소드 수집 시작
def start_episode_collection():
    # 1. 환경 초기화
    robot.reset_position()
    camera.start_recording()
    
    # 2. 태스크 선택
    task = select_random_task()
    print(f"Selected task: {task}")
    
    # 3. 에피소드 수집
    episode_data = collect_episode(task)
    
    # 4. 데이터 저장
    save_episode(episode_data)
    
    return episode_data
```

### 3.3 데이터 검증

**실시간 검증**:
```python
def validate_episode_data(episode_data):
    """에피소드 데이터 실시간 검증"""
    # 이미지 품질 검증
    for img in episode_data["images"]:
        assert img.shape == (224, 224, 3), "Invalid image shape"
        assert img.dtype == np.uint8, "Invalid image dtype"
    
    # 액션 검증
    for action in episode_data["actions"]:
        assert len(action) == 3, "Invalid action dimension"
        assert -1.0 <= action[0] <= 1.0, "X action out of range"
        assert -1.0 <= action[1] <= 1.0, "Y action out of range"
        assert action[2] in [0, 1], "Gripper action invalid"
    
    # 언어 명령 검증
    assert len(episode_data["language"]) > 0, "Empty language command"
    assert len(episode_data["language"]) < 100, "Language command too long"
    
    return True
```

## 4. 수집 일정

### 4.1 1주차: 밤 시간대 수집 (우선순위 1)

**목표**: 200개 에피소드 수집 (1주차 + 2주차)
**일정**:
- **월요일**: 20:00-22:00 (40개 에피소드)
- **화요일**: 20:00-22:00 (40개 에피소드)
- **수요일**: 20:00-22:00 (40개 에피소드)
- **목요일**: 20:00-22:00 (40개 에피소드)
- **금요일**: 20:00-22:00 (40개 에피소드)

**검증**: 매일 수집 후 데이터 품질 확인

### 4.2 2주차: 새벽 시간대 수집 (우선순위 2)

**목표**: 200개 에피소드 수집
**일정**:
- **월요일**: 00:00-04:00 (40개 에피소드)
- **화요일**: 00:00-04:00 (40개 에피소드)
- **수요일**: 00:00-04:00 (40개 에피소드)
- **목요일**: 00:00-04:00 (40개 에피소드)
- **금요일**: 00:00-04:00 (40개 에피소드)

### 4.3 3주차: 낮 시간대 수집 (최소화)

**목표**: 200개 에피소드 수집 (창 앞 테이핑 맵)
**일정**:
- **월요일**: 13:00-15:00 (40개 에피소드)
- **화요일**: 13:00-15:00 (40개 에피소드)
- **수요일**: 13:00-15:00 (40개 에피소드)
- **목요일**: 13:00-15:00 (40개 에피소드)
- **금요일**: 13:00-15:00 (40개 에피소드)

### 4.4 4주차: 추가 수집 및 검증

**목표**: 400개 에피소드 추가 수집 (밤/새벽 시간대)
**일정**: 일정한 조명 조건 우선
**검증**: 전체 데이터셋 품질 검증

## 5. 데이터 품질 관리

### 5.1 실시간 모니터링

**수집 중 모니터링**:
```python
def monitor_collection_progress():
    """수집 진행 상황 모니터링"""
    # 1. 에피소드 수 확인
    episode_count = count_episodes()
    print(f"Collected episodes: {episode_count}/1000")
    
    # 2. 데이터 품질 확인
    quality_score = check_data_quality()
    print(f"Data quality score: {quality_score}")
    
    # 3. 태스크 분포 확인
    task_distribution = check_task_distribution()
    print(f"Task distribution: {task_distribution}")
    
    return episode_count, quality_score, task_distribution
```

### 5.2 품질 검증 기준

**이미지 품질**:
- **해상도**: 224x224 픽셀
- **밝기**: 50-200 (0-255 범위)
- **대비**: 0.3 이상
- **블러**: 없음

**액션 품질**:
- **범위**: [-1, 1] (정규화됨)
- **연속성**: 급격한 변화 없음
- **물리적 가능성**: 로봇 제약 내

**언어 명령 품질**:
- **길이**: 5-50 단어
- **명확성**: 태스크 명확히 설명
- **다양성**: 반복 최소화

## 6. 다음 단계

### 6.1 즉시 실행할 작업

1. **환경 테스트**: Branch b131fb5에서 시스템 동작 확인
2. **첫 에피소드 수집**: 밤 시간대 5개 에피소드 테스트
3. **데이터 검증**: 수집된 데이터 품질 확인
4. **스크립트 최적화**: 수집 프로세스 개선

### 6.2 1주차 목표

1. **100개 에피소드 수집**: 밤 시간대 완료
2. **데이터 품질 검증**: 모든 에피소드 검증 완료
3. **수집 프로세스 최적화**: 효율성 개선
4. **다음 주 준비**: 낮 시간대 수집 준비

### 6.3 장기 목표

1. **1000개 에피소드 수집**: 3주 내 완료
2. **데이터셋 구축**: 학습용 데이터셋 완성
3. **품질 보장**: 높은 품질의 데이터셋 확보
4. **학습 준비**: LoRA Fine-tuning 준비

## 7. 참고 자료

- `RoboVLMs/core/todo_list/mobile_vla_implementation_plan.md`: 전체 구현 계획
- `RoboVLMs/core/comparison_analysis/robovlms_vs_mobile_vla.md`: 아키텍처 비교
- `RoboVLMs/core/finetuning_analysis/real_world_data_collection.md`: 데이터 수집 방법
