#!/usr/bin/env python3
"""
추론 시스템 테스트 스크립트

테스트 항목:
1. 액션 청크 형태 검증
2. 액션 값 범위 검증
3. 추론 시간 측정
4. 전체 파이프라인 테스트
"""

import unittest
import numpy as np
import torch
import time
import sys
import os

# 상위 디렉토리를 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.action_chunk_inference import (
    ActionScheduler,
    InputManager,
    InferenceValidator,
    PerformanceMonitor
)


class TestActionScheduler(unittest.TestCase):
    """ActionScheduler 테스트"""
    
    def setUp(self):
        self.scheduler = ActionScheduler(chunk_size=10, inference_interval=0.2)
    
    def test_initialization(self):
        """초기화 테스트"""
        self.assertEqual(self.scheduler.chunk_size, 10)
        self.assertEqual(self.scheduler.inference_interval, 0.2)
        self.assertFalse(self.scheduler.is_initialized)
    
    def test_should_infer_first_time(self):
        """첫 추론 필요 확인"""
        self.assertTrue(self.scheduler.should_infer(time.time()))
    
    def test_update_chunk(self):
        """청크 업데이트 테스트"""
        dummy_chunk = np.random.randn(10, 2)
        self.scheduler.update_chunk(dummy_chunk)
        
        self.assertTrue(self.scheduler.is_initialized)
        self.assertEqual(self.scheduler.chunk_index, 0)
        self.assertIsNotNone(self.scheduler.action_chunk)
    
    def test_get_current_action(self):
        """현재 액션 가져오기 테스트"""
        dummy_chunk = np.random.randn(10, 2)
        self.scheduler.update_chunk(dummy_chunk)
        
        # 10개 액션 가져오기
        for i in range(10):
            action = self.scheduler.get_current_action()
            self.assertIsNotNone(action)
            self.assertEqual(action.shape, (2,))
        
        # 11번째는 None
        action = self.scheduler.get_current_action()
        self.assertIsNone(action)
    
    def test_invalid_chunk_shape(self):
        """잘못된 청크 형태 테스트"""
        invalid_chunk = np.random.randn(5, 2)  # 잘못된 크기
        
        with self.assertRaises(AssertionError):
            self.scheduler.update_chunk(invalid_chunk)


class TestInputManager(unittest.TestCase):
    """InputManager 테스트"""
    
    def setUp(self):
        self.input_manager = InputManager(velocity_interval=0.4)
    
    def test_initialization(self):
        """초기화 테스트"""
        self.assertEqual(self.input_manager.velocity_interval, 0.4)
        self.assertEqual(len(self.input_manager.velocity_buffer), 0)
    
    def test_add_velocity(self):
        """속도 추가 테스트"""
        self.input_manager.add_velocity((0.5, 0.1))
        self.assertEqual(len(self.input_manager.velocity_buffer), 1)
        
        # 여러 개 추가
        for i in range(15):
            self.input_manager.add_velocity((0.5, 0.1))
        
        # 최대 10개만 유지
        self.assertEqual(len(self.input_manager.velocity_buffer), 10)
    
    def test_get_recent_velocities(self):
        """최근 속도 가져오기 테스트"""
        # 5개 추가
        for i in range(5):
            self.input_manager.add_velocity((i * 0.1, i * 0.05))
        
        recent = self.input_manager.get_recent_velocities(n=3)
        self.assertEqual(len(recent), 3)
    
    def test_set_initial_input(self):
        """초기 입력 설정 테스트"""
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        text = "test_command"
        distance = 1.5
        
        self.input_manager.set_initial_input(dummy_image, text, distance)
        
        self.assertIsNotNone(self.input_manager.current_image)
        self.assertEqual(self.input_manager.text_command, text)
        self.assertEqual(self.input_manager.initial_distance, distance)


class TestInferenceValidator(unittest.TestCase):
    """InferenceValidator 테스트"""
    
    def setUp(self):
        self.validator = InferenceValidator(max_linear_vel=1.0, max_angular_vel=1.0)
    
    def test_valid_action_chunk(self):
        """유효한 액션 청크 테스트"""
        valid_chunk = np.array([
            [0.5, 0.3],
            [0.4, 0.2],
            [0.3, 0.1],
            [0.2, 0.0],
            [0.1, -0.1],
            [0.0, -0.2],
            [-0.1, -0.3],
            [-0.2, -0.4],
            [-0.3, -0.5],
            [-0.4, -0.6]
        ])
        
        self.assertTrue(self.validator.validate_action_chunk(valid_chunk))
    
    def test_invalid_shape(self):
        """잘못된 형태 테스트"""
        invalid_chunk = np.random.randn(10, 3)  # 3차원
        self.assertFalse(self.validator.validate_action_chunk(invalid_chunk))
    
    def test_nan_values(self):
        """NaN 값 테스트"""
        nan_chunk = np.random.randn(10, 2)
        nan_chunk[5, 0] = np.nan
        
        self.assertFalse(self.validator.validate_action_chunk(nan_chunk))
    
    def test_inf_values(self):
        """Inf 값 테스트"""
        inf_chunk = np.random.randn(10, 2)
        inf_chunk[3, 1] = np.inf
        
        self.assertFalse(self.validator.validate_action_chunk(inf_chunk))
    
    def test_exceeds_limits(self):
        """속도 제한 초과 테스트"""
        exceed_chunk = np.random.randn(10, 2)
        exceed_chunk[0, 0] = 2.0  # 최대값 초과
        
        self.assertFalse(self.validator.validate_action_chunk(exceed_chunk))
    
    def test_log_action(self):
        """액션 로깅 테스트"""
        action = np.array([0.5, 0.3])
        timestamp = time.time()
        
        self.validator.log_action(action, timestamp)
        
        self.assertEqual(len(self.validator.action_history), 1)
        self.assertEqual(self.validator.action_history[0]['action'], action.tolist())


class TestPerformanceMonitor(unittest.TestCase):
    """PerformanceMonitor 테스트"""
    
    def setUp(self):
        self.monitor = PerformanceMonitor()
    
    def test_record_inference_time(self):
        """추론 시간 기록 테스트"""
        self.monitor.record_inference_time(50.0)
        self.monitor.record_inference_time(60.0)
        self.monitor.record_inference_time(55.0)
        
        self.assertEqual(len(self.monitor.inference_times), 3)
    
    def test_record_action_execution(self):
        """액션 실행 기록 테스트"""
        self.monitor.record_action_execution(success=True)
        self.monitor.record_action_execution(success=True)
        self.monitor.record_action_execution(success=False)
        
        self.assertEqual(self.monitor.total_actions, 3)
        self.assertEqual(self.monitor.failed_actions, 1)
    
    def test_get_statistics(self):
        """통계 계산 테스트"""
        # 추론 시간 기록
        times = [50.0, 60.0, 55.0, 45.0, 65.0]
        for t in times:
            self.monitor.record_inference_time(t)
        
        # 액션 실행 기록
        for i in range(10):
            self.monitor.record_action_execution(success=(i % 5 != 0))
        
        stats = self.monitor.get_statistics()
        
        self.assertAlmostEqual(stats['avg_inference_time_ms'], np.mean(times))
        self.assertEqual(stats['max_inference_time_ms'], max(times))
        self.assertEqual(stats['min_inference_time_ms'], min(times))
        self.assertEqual(stats['total_actions'], 10)
        self.assertEqual(stats['failed_actions'], 2)
        self.assertAlmostEqual(stats['success_rate'], 0.8)


class TestIntegration(unittest.TestCase):
    """통합 테스트"""
    
    def test_full_pipeline(self):
        """전체 파이프라인 테스트"""
        # 컴포넌트 초기화
        scheduler = ActionScheduler(chunk_size=10, inference_interval=0.2)
        input_manager = InputManager(velocity_interval=0.4)
        validator = InferenceValidator()
        monitor = PerformanceMonitor()
        
        # 초기 입력 설정
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        input_manager.set_initial_input(dummy_image, "test", 1.5)
        
        # 더미 액션 청크 생성
        dummy_chunk = np.random.uniform(-0.5, 0.5, (10, 2))
        
        # 검증
        self.assertTrue(validator.validate_action_chunk(dummy_chunk))
        
        # 스케줄러 업데이트
        scheduler.update_chunk(dummy_chunk)
        
        # 액션 실행 시뮬레이션
        for i in range(10):
            action = scheduler.get_current_action()
            self.assertIsNotNone(action)
            
            validator.log_action(action, time.time())
            monitor.record_action_execution(success=True)
            monitor.record_inference_time(50.0)
        
        # 통계 확인
        stats = monitor.get_statistics()
        self.assertEqual(stats['total_actions'], 10)
        self.assertEqual(stats['failed_actions'], 0)
        self.assertEqual(stats['success_rate'], 1.0)


def run_tests():
    """테스트 실행"""
    print("="*60)
    print("Running Inference System Tests")
    print("="*60)
    
    # 테스트 스위트 생성
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 테스트 추가
    suite.addTests(loader.loadTestsFromTestCase(TestActionScheduler))
    suite.addTests(loader.loadTestsFromTestCase(TestInputManager))
    suite.addTests(loader.loadTestsFromTestCase(TestInferenceValidator))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformanceMonitor))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # 테스트 실행
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 결과 출력
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.2f}%")
    print("="*60)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
