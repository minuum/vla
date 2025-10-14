# RoboVLMs 평가 과정 분석 (한글)

## 평가 프레임워크 개요

RoboVLMs 프레임워크는 CALVIN, SimplerEnv, 실제 로봇 실험을 포함한 여러 벤치마크에 대한 포괄적인 평가 시스템을 제공합니다. 평가 과정은 엄격하고 재현 가능하며 포괄적으로 설계되었습니다.

## CALVIN 평가

### 1. 평가 설정
```python
class CalvinEvaluator:
    """
    CALVIN 벤치마크 평가
    """
    
    def __init__(self, config):
        self.config = config
        self.env = self.setup_environment()
        self.metrics = self.setup_metrics()
    
    def setup_environment(self):
        """CALVIN 환경 설정"""
        env = CalvinEnv(
            data_path=self.config.data_path,
            split=self.config.split,
            task_length=self.config.task_length
        )
        return env
    
    def setup_metrics(self):
        """평가 메트릭 설정"""
        return {
            'success_rate': SuccessRateMetric(),
            'avg_length': AverageLengthMetric(),
            'generalization': GeneralizationMetric()
        }
```

### 2. 평가 과정
```python
def evaluate_model(self, model, test_loader):
    """
    CALVIN 벤치마크에서 모델 평가
    """
    results = {
        'consecutive_tasks': {},
        'avg_length': 0.0,
        'generalization_score': 0.0
    }
    
    for batch in test_loader:
        # 평가 실행
        batch_results = self.run_evaluation(model, batch)
        
        # 결과 업데이트
        for task_length in range(1, 6):
            if task_length not in results['consecutive_tasks']:
                results['consecutive_tasks'][task_length] = []
            results['consecutive_tasks'][task_length].append(
                batch_results[f'task_{task_length}_success']
            )
        
        results['avg_length'] += batch_results['avg_length']
        results['generalization_score'] += batch_results['generalization']
    
    # 평균 계산
    for task_length in results['consecutive_tasks']:
        results['consecutive_tasks'][task_length] = np.mean(
            results['consecutive_tasks'][task_length]
        )
    
    results['avg_length'] /= len(test_loader)
    results['generalization_score'] /= len(test_loader)
    
    return results
```

### 3. 성공률 계산
```python
def calculate_success_rate(self, predictions, targets):
    """
    연속 작업에 대한 성공률 계산
    """
    success_rates = {}
    
    for task_length in range(1, 6):
        # 성공적인 연속 작업 계산
        successful_tasks = 0
        total_tasks = 0
        
        for pred, target in zip(predictions, targets):
            if len(pred) >= task_length and len(target) >= task_length:
                # 첫 번째 task_length 작업이 성공적인지 확인
                if self.check_consecutive_success(pred[:task_length], target[:task_length]):
                    successful_tasks += 1
                total_tasks += 1
        
        success_rates[task_length] = successful_tasks / total_tasks if total_tasks > 0 else 0.0
    
    return success_rates
```

### 4. 평균 길이 계산
```python
def calculate_avg_length(self, predictions, targets):
    """
    완료된 작업의 평균 수 계산
    """
    total_length = 0
    total_episodes = 0
    
    for pred, target in zip(predictions, targets):
        # 완료된 작업 계산
        completed_tasks = self.count_completed_tasks(pred, target)
        total_length += completed_tasks
        total_episodes += 1
    
    return total_length / total_episodes if total_episodes > 0 else 0.0
```

## SimplerEnv 평가

### 1. Google Robot 평가
```python
class GoogleRobotEvaluator:
    """
    Google Robot 환경 평가
    """
    
    def __init__(self, config):
        self.config = config
        self.env = self.setup_environment()
        self.tasks = self.setup_tasks()
    
    def setup_tasks(self):
        """Google Robot 작업 설정"""
        tasks = {
            'pick_coke_can': PickCokeCanTask(),
            'move_near': MoveNearTask(),
            'open_close_drawer': OpenCloseDrawerTask(),
            'open_drawer_place_apple': OpenDrawerPlaceAppleTask()
        }
        return tasks
    
    def evaluate_task(self, model, task):
        """
        특정 작업에서 모델 평가
        """
        success_count = 0
        total_trials = 0
        
        for trial in task.trials:
            # 시행 실행
            success = self.run_trial(model, task, trial)
            if success:
                success_count += 1
            total_trials += 1
        
        return success_count / total_trials if total_trials > 0 else 0.0
```

### 2. WidowX+Bridge 평가
```python
class WidowXBridgeEvaluator:
    """
    WidowX+Bridge 환경 평가
    """
    
    def __init__(self, config):
        self.config = config
        self.env = self.setup_environment()
        self.tasks = self.setup_tasks()
    
    def setup_tasks(self):
        """WidowX+Bridge 작업 설정"""
        tasks = {
            'put_spoon_on_towel': PutSpoonOnTowelTask(),
            'put_carrot_on_plate': PutCarrotOnPlateTask(),
            'stack_green_block_on_yellow_block': StackGreenBlockOnYellowBlockTask(),
            'put_eggplant_in_yellow_basket': PutEggplantInYellowBasketTask()
        }
        return tasks
    
    def evaluate_task(self, model, task):
        """
        특정 작업에서 모델 평가
        """
        success_count = 0
        total_trials = 0
        
        for trial in task.trials:
            # 시행 실행
            success = self.run_trial(model, task, trial)
            if success:
                success_count += 1
            total_trials += 1
        
        return success_count / total_trials if total_trials > 0 else 0.0
```

## 실제 평가

### 1. 실제 로봇 설정
```python
class RealWorldEvaluator:
    """
    실제 로봇 평가
    """
    
    def __init__(self, config):
        self.config = config
        self.robot = self.setup_robot()
        self.cameras = self.setup_cameras()
        self.tasks = self.setup_tasks()
    
    def setup_robot(self):
        """실제 로봇 플랫폼 설정"""
        robot = KinovaGen3(
            arm_config=self.config.arm_config,
            gripper_config=self.config.gripper_config
        )
        return robot
    
    def setup_cameras(self):
        """카메라 시스템 설정"""
        cameras = {
            'workspace': KinectAzure(),
            'wrist': RealSenseD435i()
        }
        return cameras
    
    def setup_tasks(self):
        """실제 작업 설정"""
        tasks = {
            'open_drawer': OpenDrawerTask(),
            'pickup_eggplant': PickupEggplantTask(),
            'press_toaster': PressToasterTask(),
            'pickup_knife': PickupKnifeTask(),
            'pickup_cucumber': PickupCucumberTask()
        }
        return tasks
```

### 2. 평가 설정
```python
def setup_evaluation_settings(self):
    """
    평가 설정 설정
    """
    settings = {
        'simple': SimpleSetting(),
        'unseen_distractor': UnseenDistractorSetting(),
        'unseen_background': UnseenBackgroundSetting(),
        'unseen_object': UnseenObjectSetting(),
        'novel_skill_description': NovelSkillDescriptionSetting()
    }
    return settings
```

### 3. 작업 평가
```python
def evaluate_task(self, model, task, setting):
    """
    특정 작업 및 설정에서 모델 평가
    """
    success_count = 0
    total_rollouts = 0
    
    for rollout in range(5):  # 작업당 5회 롤아웃
        # 환경 설정
        env = self.setup_environment(task, setting)
        
        # 롤아웃 실행
        success = self.run_rollout(model, env, task)
        if success:
            success_count += 1
        total_rollouts += 1
    
    return success_count / total_rollouts if total_rollouts > 0 else 0.0
```

## 평가 메트릭

### 1. 성공률 메트릭
```python
class SuccessRateMetric:
    """
    성공률 계산
    """
    
    def __init__(self):
        self.success_count = 0
        self.total_count = 0
    
    def update(self, predictions, targets):
        """성공률 업데이트"""
        for pred, target in zip(predictions, targets):
            if self.check_success(pred, target):
                self.success_count += 1
            self.total_count += 1
    
    def compute(self):
        """성공률 계산"""
        return self.success_count / self.total_count if self.total_count > 0 else 0.0
```

### 2. 평균 길이 메트릭
```python
class AverageLengthMetric:
    """
    평균 길이 계산
    """
    
    def __init__(self):
        self.total_length = 0
        self.total_episodes = 0
    
    def update(self, predictions, targets):
        """평균 길이 업데이트"""
        for pred, target in zip(predictions, targets):
            length = self.count_completed_tasks(pred, target)
            self.total_length += length
            self.total_episodes += 1
    
    def compute(self):
        """평균 길이 계산"""
        return self.total_length / self.total_episodes if self.total_episodes > 0 else 0.0
```

### 3. 일반화 메트릭
```python
class GeneralizationMetric:
    """
    일반화 점수 계산
    """
    
    def __init__(self):
        self.generalization_scores = []
    
    def update(self, predictions, targets, unseen_ratio):
        """일반화 점수 업데이트"""
        score = self.compute_generalization_score(predictions, targets, unseen_ratio)
        self.generalization_scores.append(score)
    
    def compute(self):
        """평균 일반화 점수 계산"""
        return np.mean(self.generalization_scores) if self.generalization_scores else 0.0
```

## 자기 수정 평가

### 1. 자기 수정 감지
```python
def detect_self_correction(self, trajectory, target):
    """
    궤적에서 자기 수정 행동 감지
    """
    corrections = []
    
    for i in range(1, len(trajectory)):
        # 현재 액션이 이전 액션을 수정하는지 확인
        if self.is_correction(trajectory[i-1], trajectory[i], target):
            corrections.append(i)
    
    return corrections
```

### 2. 자기 수정 분석
```python
def analyze_self_correction(self, model, test_cases):
    """
    자기 수정 능력 분석
    """
    correction_results = {
        'correction_rate': 0.0,
        'correction_effectiveness': 0.0,
        'correction_timing': []
    }
    
    for test_case in test_cases:
        # 궤적 실행
        trajectory = self.run_trajectory(model, test_case)
        
        # 수정 감지
        corrections = self.detect_self_correction(trajectory, test_case.target)
        
        # 수정 분석
        if corrections:
            correction_results['correction_rate'] += 1
            effectiveness = self.measure_correction_effectiveness(
                trajectory, corrections, test_case.target
            )
            correction_results['correction_effectiveness'] += effectiveness
            correction_results['correction_timing'].extend(corrections)
    
    # 평균 계산
    correction_results['correction_rate'] /= len(test_cases)
    correction_results['correction_effectiveness'] /= len(test_cases)
    
    return correction_results
```

## 평가 파이프라인

### 1. 자동화된 평가
```python
def run_automated_evaluation(self, model, config):
    """
    자동화된 평가 파이프라인 실행
    """
    results = {}
    
    # CALVIN 평가
    if 'calvin' in config.benchmarks:
        calvin_results = self.run_calvin_evaluation(model, config.calvin)
        results['calvin'] = calvin_results
    
    # SimplerEnv 평가
    if 'simplerenv' in config.benchmarks:
        simplerenv_results = self.run_simplerenv_evaluation(model, config.simplerenv)
        results['simplerenv'] = simplerenv_results
    
    # 실제 평가
    if 'real_world' in config.benchmarks:
        real_world_results = self.run_real_world_evaluation(model, config.real_world)
        results['real_world'] = real_world_results
    
    return results
```

### 2. 평가 보고서 생성
```python
def generate_evaluation_report(self, results):
    """
    포괄적인 평가 보고서 생성
    """
    report = {
        'summary': self.generate_summary(results),
        'detailed_results': results,
        'comparisons': self.generate_comparisons(results),
        'recommendations': self.generate_recommendations(results)
    }
    
    return report
```

### 3. 성능 비교
```python
def compare_performance(self, results, baselines):
    """
    베이스라인과 성능 비교
    """
    comparisons = {}
    
    for benchmark in results:
        benchmark_results = results[benchmark]
        baseline_results = baselines[benchmark]
        
        comparisons[benchmark] = {
            'improvement': self.calculate_improvement(benchmark_results, baseline_results),
            'relative_performance': self.calculate_relative_performance(benchmark_results, baseline_results)
        }
    
    return comparisons
```

## 평가 모범 사례

### 1. 재현성
```python
def ensure_reproducibility(self, config):
    """
    평가 재현성 보장
    """
    # 랜덤 시드 설정
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    
    # 결정적 동작 설정
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

### 2. 통계적 유의성
```python
def calculate_statistical_significance(self, results, baseline_results):
    """
    결과의 통계적 유의성 계산
    """
    from scipy import stats
    
    # t-검정 수행
    t_stat, p_value = stats.ttest_ind(results, baseline_results)
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    }
```

### 3. 오류 분석
```python
def analyze_errors(self, predictions, targets):
    """
    예측 오류 분석
    """
    errors = {
        'pose_errors': [],
        'gripper_errors': [],
        'timing_errors': []
    }
    
    for pred, target in zip(predictions, targets):
        # 포즈 오류 계산
        pose_error = np.linalg.norm(pred[:6] - target[:6])
        errors['pose_errors'].append(pose_error)
        
        # 그리퍼 오류 계산
        gripper_error = abs(pred[6] - target[6])
        errors['gripper_errors'].append(gripper_error)
        
        # 타이밍 오류 계산
        timing_error = abs(len(pred) - len(target))
        errors['timing_errors'].append(timing_error)
    
    return errors
```

## 결론

RoboVLMs 평가 과정은 여러 벤치마크와 시나리오에서 VLA 모델을 평가하기 위한 포괄적인 프레임워크를 제공합니다. 평가 시스템은 다음과 같은 특징을 포함합니다:

### 주요 특징
1. **포괄적인 벤치마크**: CALVIN, SimplerEnv, 실제 평가
2. **다양한 메트릭**: 성공률, 평균 길이, 일반화 점수
3. **자기 수정 분석**: 자기 수정 행동의 감지 및 분석
4. **통계적 엄격성**: 통계적 유의성 테스트 및 오류 분석
5. **재현성**: 적절한 시딩을 통한 결정적 평가

### 평가 이점
1. **엄격한 평가**: 여러 시나리오에서 포괄적인 평가
2. **성능 비교**: 베이스라인 방법과의 상세한 비교
3. **오류 분석**: 예측 오류의 심층 분석
4. **일반화 테스트**: 모델 일반화 능력 평가
5. **실제 검증**: 실제 로봇 플랫폼에서의 검증
