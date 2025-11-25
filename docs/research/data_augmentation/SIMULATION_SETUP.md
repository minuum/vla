# Mobile-VLA 시뮬레이션 환경 구축 가이드

## 🎯 최종 선택: Habitat-AI

### 선정 이유

| 기준 | Habitat-AI | Isaac Sim | PyBullet |
|------|-----------|-----------|----------|
| **설치 난이도** | ⭐⭐ 보통 | ⭐⭐⭐⭐ 높음 | ⭐ 쉬움 |
| **내비게이션 최적화** | ✅ **매우 우수** | ⭐⭐ 보통 | ⭐ 제한적 |
| **실내 환경 데이터** | ✅ **풍부** (Matterport3D) | 자체 제작 필요 | 자체 제작 필요 |
| **렌더링 품질** | ⭐⭐⭐ 우수 | ⭐⭐⭐⭐ 최고 | ⭐ 낮음 |
| **Python API** | ✅ **직관적** | 복잡 | 직관적 |
| **GPU 병렬화** | ⭐⭐ 지원 | ⭐⭐⭐⭐ 최고 | ⭐ 제한적 |
| **커뮤니티/문서** | ✅ **활발** | 중간 | 활발 |
| **Mobile Robot 사례** | ✅ **많음** | 있음 | 있음 |

**결론**: Mobile-VLA는 **실내 내비게이션**에 특화된 프로젝트이므로, 실제 건물 스캔 데이터를 활용할 수 있고 내비게이션 태스크에 최적화된 **Habitat-AI**가 최적의 선택입니다.

---

## 📦 설치 가이드

### 1. 시스템 요구사항

```yaml
OS: Ubuntu 20.04 / 22.04 (권장), macOS (제한적 지원)
Python: 3.9 - 3.10
GPU: NVIDIA GPU (CUDA 11.0+) - 선택사항이지만 강력 권장
RAM: 16GB 이상
Storage: 50GB 이상 (씬 데이터 포함)
```

### 2. Conda 환경 생성

```bash
# Conda 환경 생성
conda create -n habitat python=3.10 cmake=3.14.0 -y
conda activate habitat

# 기본 의존성 설치
conda install habitat-sim headless -c conda-forge -c aihabitat -y
```

### 3. Habitat-Lab 설치

```bash
# Habitat-Lab 클론 및 설치
git clone --branch stable https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab

# 개발 모드로 설치
pip install -e habitat-lab
pip install -e habitat-baselines

# 추가 의존성
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python matplotlib numpy scipy h5py pyyaml tqdm
```

### 4. 씬 데이터 다운로드

```bash
# Matterport3D 씬 다운로드 (약 15GB)
# 학술 목적 라이센스 필요: https://niessner.github.io/Matterport/
python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes --data-path data/

# Gibson 씬 (대안)
python -m habitat_sim.utils.datasets_download --uids habitat_test_pointnav_dataset --data-path data/
```

### 5. 설치 검증

```bash
# Habitat-Sim 테스트
python -c "import habitat_sim; print('Habitat-Sim version:', habitat_sim.__version__)"

# Habitat-Lab 테스트
python -c "import habitat; print('Habitat-Lab version:', habitat.__version__)"

# 간단한 시뮬레이션 실행
python examples/tutorials/habitat_lab_demo.py
```

---

## 🛠️ Mobile-VLA 통합 설정

### 디렉토리 구조

```bash
# 프로젝트 루트에 시뮬레이션 디렉토리 생성
cd /path/to/vla
mkdir -p simulation/{environments,configs,data_generator,utils}
```

**구조**:
```
simulation/
├── environments/          # 환경 정의
│   ├── __init__.py
│   ├── office_nav_env.py # 사무실 내비게이션 환경
│   └── hallway_nav_env.py # 복도 내비게이션 환경
├── configs/               # 설정 파일
│   ├── default.yaml      # 기본 설정
│   └── domain_randomization.yaml
├── data_generator/        # 데이터 생성기
│   ├── __init__.py
│   ├── trajectory_generator.py
│   └── h5_exporter.py    # Mobile-VLA 형식 변환
├── utils/                 # 유틸리티
│   ├── __init__.py
│   └── visualization.py  # 궤적 시각화
└── README.md
```

### 환경 설정 파일 생성

**`simulation/configs/default.yaml`**:
```yaml
# Habitat-AI 기본 설정
habitat:
  simulator:
    type: "Sim-v0"
    action_space_config: "v0"
    forward_step_size: 0.25  # 이동 크기 (미터)
    turn_angle: 10           # 회전 각도 (도)
    
  task:
    type: PointNav-v0
    sensors:
      - type: RGBSensor
        height: 224
        width: 224
        position: [0, 1.25, 0]  # 카메라 높이
        orientation: [0, 0, 0]
    
    measurements:
      - type: TopDownMap
      - type: DistanceToGoal
      - type: Success
      - type: SPL  # Success weighted by Path Length
    
    goal_sensor_uuid: pointgoal_with_gps_compass
    
# Mobile-VLA 특화 설정
mobile_vla:
  robot:
    base_height: 0.5        # 로봇 베이스 높이 (m)
    max_linear_velocity: 1.0 # 최대 선속도 (m/s)
    max_angular_velocity: 1.5 # 최대 각속도 (rad/s)
    
  data_collection:
    episodes_per_scene: 50   # 씬당 에피소드 수
    max_steps: 500          # 에피소드당 최대 스텝
    success_distance: 0.2   # 성공 거리 (m)
    
  domain_randomization:
    enabled: true
    lighting_range: [0.5, 1.5]
    camera_tilt_range: [-5, 5]  # 도
    
# 저장 경로
output:
  dataset_dir: "./data/synthetic_episodes"
  format: "h5"
  compress: true
```

---

## 🚀 첫 번째 환경 구현

### `simulation/environments/office_nav_env.py`

```python
"""
Mobile-VLA를 위한 Habitat-AI 사무실 내비게이션 환경
"""
import habitat
from habitat.config.default import get_config
from habitat.core.env import Env
import numpy as np
import quaternion
from typing import Dict, Optional, Tuple

class OfficeNavigationEnv:
    """
    사무실 환경 내비게이션 시뮬레이터
    Mobile-VLA 데이터 포맷 (2DOF: linear_x, angular_z) 생성
    """
    
    def __init__(self, config_path: str = "simulation/configs/default.yaml"):
        # Habitat 설정 로드
        self.config = get_config(config_path)
        self.env: Optional[Env] = None
        self.current_episode = 0
        
    def reset(self) -> Dict[str, np.ndarray]:
        """환경 리셋 및 초기 관측 반환"""
        if self.env is None:
            self.env = habitat.Env(config=self.config)
        
        observations = self.env.reset()
        self.current_episode += 1
        
        return {
            'rgb': observations['rgb'],  # (224, 224, 3)
            'goal_position': self._get_goal_position(),
            'robot_position': self._get_robot_position()
        }
    
    def step(self, action: Dict[str, float]) -> Tuple[Dict, float, bool, Dict]:
        """
        Mobile-VLA 액션 적용 (2DOF)
        
        Args:
            action: {'linear_x': 속도(m/s), 'angular_z': 각속도(rad/s)}
        
        Returns:
            observation, reward, done, info
        """
        # 2DOF 속도를 Habitat 이산 액션으로 변환
        habitat_action = self._velocity_to_habitat_action(
            action['linear_x'], 
            action['angular_z']
        )
        
        observations = self.env.step(habitat_action)
        
        # 보상 계산
        metrics = self.env.get_metrics()
        reward = self._compute_reward(metrics)
        
        done = self.env.episode_over
        
        info = {
            'distance_to_goal': metrics.get('distance_to_goal', -1),
            'success': metrics.get('success', 0),
            'spl': metrics.get('spl', 0)
        }
        
        return observations, reward, done, info
    
    def _velocity_to_habitat_action(self, linear: float, angular: float) -> int:
        """
        2DOF 속도 명령을 Habitat 이산 액션으로 매핑
        
        Habitat Actions:
        0: STOP
        1: MOVE_FORWARD
        2: TURN_LEFT
        3: TURN_RIGHT
        """
        # 임계값 설정
        lin_threshold = 0.1  # m/s
        ang_threshold = 0.2  # rad/s
        
        if abs(linear) < lin_threshold and abs(angular) < ang_threshold:
            return 0  # STOP
        
        if abs(angular) > abs(linear):
            # 회전 우선
            return 2 if angular > 0 else 3  # LEFT or RIGHT
        else:
            # 전진 우선
            return 1 if linear > 0 else 0  # FORWARD or STOP
    
    def _get_goal_position(self) -> np.ndarray:
        """목표 위치 가져오기 (x, y, z)"""
        if self.env and self.env.current_episode:
            goals = self.env.current_episode.goals
            if goals:
                return np.array(goals[0].position)
        return np.zeros(3)
    
    def _get_robot_position(self) -> np.ndarray:
        """로봇 현재 위치 가져오기 (x, y, z)"""
        if self.env:
            agent_state = self.env.sim.get_agent_state()
            return agent_state.position
        return np.zeros(3)
    
    def _compute_reward(self, metrics: Dict) -> float:
        """
        보상 함수 정의
        - 목표에 가까워지면 양의 보상
        - 충돌 시 음의 보상
        - 목표 도달 시 큰 보상
        """
        reward = 0.0
        
        # 목표까지의 거리 기반 보상
        dist = metrics.get('distance_to_goal', 0)
        if hasattr(self, '_prev_distance'):
            reward += (self._prev_distance - dist) * 10.0  # 가까워지면 양수
        self._prev_distance = dist
        
        # 성공 보상
        if metrics.get('success', 0) > 0:
            reward += 100.0
        
        # 시간 페널티 (빨리 도달 유도)
        reward -= 0.01
        
        return reward
    
    def close(self):
        """환경 종료"""
        if self.env:
            self.env.close()
```

---

## 🧪 테스트 스크립트

### `simulation/test_env.py`

```python
"""
Habitat-AI 환경 테스트 스크립트
"""
import numpy as np
from environments.office_nav_env import OfficeNavigationEnv
import matplotlib.pyplot as plt

def test_basic_navigation():
    """기본 내비게이션 테스트"""
    print("🧪 Habitat-AI 환경 테스트 시작...")
    
    # 환경 초기화
    env = OfficeNavigationEnv()
    
    # 에피소드 실행
    num_episodes = 3
    
    for ep in range(num_episodes):
        print(f"\n📍 Episode {ep + 1}/{num_episodes}")
        
        obs = env.reset()
        done = False
        step_count = 0
        total_reward = 0
        
        # 간단한 랜덤 정책
        while not done and step_count < 100:
            # 랜덤 액션 생성
            action = {
                'linear_x': np.random.uniform(-0.5, 1.0),
                'angular_z': np.random.uniform(-1.0, 1.0)
            }
            
            obs, reward, done, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            if step_count % 10 == 0:
                print(f"  Step {step_count}: "
                      f"Dist={info['distance_to_goal']:.2f}m, "
                      f"Reward={reward:.2f}")
        
        print(f"✅ Episode finished: "
              f"Steps={step_count}, "
              f"Total Reward={total_reward:.2f}, "
              f"Success={info['success']}")
    
    env.close()
    print("\n✅ 테스트 완료!")

if __name__ == "__main__":
    test_basic_navigation()
```

---

## 📅 다음 단계

### 1주차: 환경 검증 (현재)
- [x] Habitat-AI 설치
- [x] 기본 환경 클래스 구현
- [ ] 테스트 스크립트 실행 및 검증

### 2주차: Domain Randomization
- [ ] 조명/텍스처 랜덤화 구현
- [ ] 장애물 자동 배치
- [ ] 다양한 씬 로드

### 3주차: 데이터 생성 파이프라인
- [ ] 자동 궤적 수집기 구현
- [ ] H5 포맷 변환기 구현
- [ ] 첫 100개 에피소드 생성

### 4주차: 검증 및 통합
- [ ] 생성 데이터 품질 검증
- [ ] Mobile-VLA 학습 파이프라인 통합
- [ ] 성능 비교 (synthetic vs real)

---

## 🔧 트러블슈팅

### 문제 1: CUDA 관련 오류
```bash
# GPU 없이 CPU 모드로 실행
export HABITAT_SIM_LOG=quiet
export MAGNUM_LOG=quiet
habitat-viewer --no-display
```

### 문제 2: 씬 데이터 접근 오류
```bash
# 데이터 경로 확인
ls -la data/scene_datasets/
# 환경 변수 설정
export HABITAT_DATA_PATH=/path/to/habitat-lab/data
```

### 문제 3: macOS에서 실행 시 렌더링 문제
- macOS는 headless 모드 지원이 제한적
- Linux 환경 또는 Google Colab 사용 권장

---

**업데이트**: 2025-11-25  
**다음 문서**: `DATA_GENERATION_PIPELINE.md` (데이터 생성 파이프라인 상세)
