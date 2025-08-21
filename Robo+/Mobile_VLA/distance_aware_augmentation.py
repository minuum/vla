#!/usr/bin/env python3
"""
📏 거리별 특화 데이터 증강
"""
import sys
from pathlib import Path
import numpy as np
import h5py
import json
import logging
from datetime import datetime
from PIL import Image
import random
from typing import Dict, List, Tuple
import cv2

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DistanceAwareAugmentation:
    def __init__(self):
        self.DATA_DIR = Path("/home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset")
        self.OUTPUT_DIR = Path("distance_aware_augmented_dataset")
        self.OUTPUT_DIR.mkdir(exist_ok=True)
        
        # 거리별 증강 배수 (실제 데이터 분포에 맞춤)
        self.distance_augmentation_factors = {
            "close": 8,     # 정밀 조작 강화 (20개 → 160개)
            "medium": 5,    # 표준 패턴 다양화 (32개 → 160개)
            "far": 8        # 넓은 움직임 강화 (20개 → 160개)
        }
        
        # 거리별 특화 증강 전략
        self.distance_strategies = {
            "close": {
                "description": "정밀 조작 강화",
                "noise_range": (0.05, 0.15),  # 작은 노이즈
                "space_frequency": 0.3,       # 정지 액션 빈도 증가
                "horizontal_weight": 1.5,     # A/D 액션 강화
                "precision_focus": True
            },
            "medium": {
                "description": "표준 패턴 다양화",
                "noise_range": (0.1, 0.25),   # 중간 노이즈
                "space_frequency": 0.2,       # 표준 정지 빈도
                "diagonal_weight": 1.3,       # Q/E 액션 강화
                "timing_variation": True
            },
            "far": {
                "description": "넓은 움직임 강화",
                "noise_range": (0.15, 0.3),   # 큰 노이즈
                "space_frequency": 0.1,       # 낮은 정지 빈도
                "forward_weight": 1.4,        # W 액션 강화
                "speed_variation": True
            }
        }
        
        # 실제 사용된 액션 (회전, 후진, Z/C 미사용)
        self.actual_actions = {
            'w': {"linear_x": 1.15, "linear_y": 0.0, "angular_z": 0.0},      # 전진
            'a': {"linear_x": 0.0, "linear_y": 1.15, "angular_z": 0.0},      # 좌측 이동
            'd': {"linear_x": 0.0, "linear_y": -1.15, "angular_z": 0.0},     # 우측 이동
            'q': {"linear_x": 1.15, "linear_y": 1.15, "angular_z": 0.0},     # 전진+좌측
            'e': {"linear_x": 1.15, "linear_y": -1.15, "angular_z": 0.0},    # 전진+우측
            ' ': {"linear_x": 0.0, "linear_y": 0.0, "angular_z": 0.0}        # 정지
        }

    def extract_distance_from_filename(self, filename: str) -> str:
        """파일명에서 거리 정보 추출"""
        if "_close" in filename:
            return "close"
        elif "_far" in filename:
            return "far"
        else:
            return "medium"  # 기본값

    def load_original_dataset(self) -> List[Dict]:
        """원본 데이터셋 로드"""
        episodes = []
        
        # .h5 파일들 처리 (legacy 제외)
        h5_files = [f for f in self.DATA_DIR.glob("*.h5") if "legacy" not in str(f)]
        logger.info(f"📁 발견된 .h5 파일: {len(h5_files)}개 (legacy 제외)")
        
        for h5_file in h5_files:
            try:
                with h5py.File(h5_file, 'r') as f:
                    episode_data = {
                        'images': [],
                        'actions': [],
                        'episode_id': h5_file.stem,
                        'distance': self.extract_distance_from_filename(h5_file.stem)
                    }
                    
                    # 이미지 데이터 로드
                    if 'images' in f:
                        images_data = f['images'][:]
                        for img_data in images_data:
                            if img_data.dtype == np.uint8:
                                image = Image.fromarray(img_data)
                            else:
                                img_data = (img_data * 255).astype(np.uint8)
                                image = Image.fromarray(img_data)
                            episode_data['images'].append(image)
                    
                    # 액션 데이터 로드
                    if 'actions' in f:
                        actions = f['actions'][:]
                        episode_data['actions'] = actions
                    
                    if len(episode_data['images']) > 0 and len(episode_data['actions']) > 0:
                        episodes.append(episode_data)
                        logger.info(f"✅ {h5_file.name} 로드 완료: {len(episode_data['images'])} 프레임, 거리: {episode_data['distance']}")
                        
            except Exception as e:
                logger.warning(f"⚠️ {h5_file.name} 로드 실패: {e}")
                continue
        
        return episodes

    def apply_distance_specific_augmentation(self, episode_data: Dict, distance: str) -> List[Dict]:
        """거리별 특화 증강 적용"""
        strategy = self.distance_strategies[distance]
        augmentation_factor = self.distance_augmentation_factors[distance]
        
        augmented_episodes = []
        
        for i in range(augmentation_factor - 1):  # 원본 제외
            augmented_episode = self._create_augmented_episode(episode_data, strategy, i)
            augmented_episodes.append(augmented_episode)
        
        return augmented_episodes

    def _create_augmented_episode(self, original_episode: Dict, strategy: Dict, aug_index: int) -> Dict:
        """개별 증강 에피소드 생성"""
        images = original_episode['images'].copy()
        actions = original_episode['actions'].copy()
        
        # 거리별 특화 증강 적용
        if strategy.get('precision_focus'):
            # Close 거리: 정밀 조작 강화
            actions = self._apply_precision_augmentation(actions, strategy)
        elif strategy.get('timing_variation'):
            # Medium 거리: 타이밍 변화
            actions = self._apply_timing_augmentation(actions, strategy)
        elif strategy.get('speed_variation'):
            # Far 거리: 속도 변화
            actions = self._apply_speed_augmentation(actions, strategy)
        
        # 이미지 증강 (거리별 특화)
        images = self._apply_distance_specific_image_augmentation(images, strategy)
        
        return {
            'images': images,
            'actions': actions,
            'episode_id': f"{original_episode['episode_id']}_aug_{aug_index:03d}",
            'distance': original_episode['distance'],
            'augmentation_type': strategy['description'],
            'original_episode': original_episode['episode_id']
        }

    def _apply_precision_augmentation(self, actions: np.ndarray, strategy: Dict) -> np.ndarray:
        """정밀 조작 강화 (Close 거리)"""
        noise_range = strategy['noise_range']
        space_freq = strategy['space_frequency']
        horizontal_weight = strategy['horizontal_weight']
        
        augmented_actions = actions.copy()
        
        for i in range(len(augmented_actions)):
            # 작은 노이즈 추가
            noise = np.random.uniform(-noise_range[0], noise_range[0], 3)
            augmented_actions[i] += noise
            
            # 정지 액션 빈도 증가
            if random.random() < space_freq:
                augmented_actions[i] = np.array([0.0, 0.0, 0.0])
            
            # A/D 액션 강화 (횡이동)
            if abs(augmented_actions[i][1]) > 0.1:  # linear_y가 있는 경우
                augmented_actions[i][1] *= horizontal_weight
        
        return augmented_actions

    def _apply_timing_augmentation(self, actions: np.ndarray, strategy: Dict) -> np.ndarray:
        """타이밍 변화 (Medium 거리)"""
        noise_range = strategy['noise_range']
        diagonal_weight = strategy['diagonal_weight']
        
        augmented_actions = actions.copy()
        
        for i in range(len(augmented_actions)):
            # 중간 노이즈 추가
            noise = np.random.uniform(-noise_range[0], noise_range[0], 3)
            augmented_actions[i] += noise
            
            # Q/E 액션 강화 (대각선)
            if abs(augmented_actions[i][0]) > 0.1 and abs(augmented_actions[i][1]) > 0.1:
                augmented_actions[i][0] *= diagonal_weight
                augmented_actions[i][1] *= diagonal_weight
        
        return augmented_actions

    def _apply_speed_augmentation(self, actions: np.ndarray, strategy: Dict) -> np.ndarray:
        """속도 변화 (Far 거리)"""
        noise_range = strategy['noise_range']
        forward_weight = strategy['forward_weight']
        
        augmented_actions = actions.copy()
        
        for i in range(len(augmented_actions)):
            # 큰 노이즈 추가
            noise = np.random.uniform(-noise_range[0], noise_range[0], 3)
            augmented_actions[i] += noise
            
            # W 액션 강화 (전진)
            if augmented_actions[i][0] > 0.1:  # linear_x가 양수인 경우
                augmented_actions[i][0] *= forward_weight
        
        return augmented_actions

    def _apply_distance_specific_image_augmentation(self, images: List[Image.Image], strategy: Dict) -> List[Image.Image]:
        """거리별 특화 이미지 증강"""
        augmented_images = []
        
        for image in images:
            # 기본 이미지 변환
            img_array = np.array(image)
            
            # 거리별 특화 변환
            if strategy.get('precision_focus'):
                # Close: 미세한 밝기 조정
                brightness = np.random.uniform(0.95, 1.05)
                img_array = np.clip(img_array * brightness, 0, 255).astype(np.uint8)
            elif strategy.get('timing_variation'):
                # Medium: 중간 정도의 대비 조정
                contrast = np.random.uniform(0.9, 1.1)
                img_array = np.clip((img_array - 128) * contrast + 128, 0, 255).astype(np.uint8)
            elif strategy.get('speed_variation'):
                # Far: 큰 변화 (블러 효과)
                if random.random() < 0.3:
                    kernel_size = random.choice([3, 5])
                    img_array = cv2.GaussianBlur(img_array, (kernel_size, kernel_size), 0)
            
            augmented_images.append(Image.fromarray(img_array))
        
        return augmented_images

    def save_augmented_episode(self, episode_data: Dict, episode_dir: Path):
        """증강된 에피소드 저장"""
        episode_dir.mkdir(exist_ok=True)
        
        # 이미지 저장
        for i, image in enumerate(episode_data['images']):
            image_path = episode_dir / f"frame_{i:03d}.jpg"
            image.save(image_path, "JPEG", quality=95)
        
        # 액션 저장
        actions_path = episode_dir / "actions.npy"
        np.save(actions_path, episode_data['actions'])
        
        # 메타데이터 저장
        metadata = {
            "episode_id": episode_data['episode_id'],
            "distance": episode_data['distance'],
            "augmentation_type": episode_data.get('augmentation_type', 'original'),
            "original_episode": episode_data.get('original_episode', episode_data['episode_id']),
            "num_frames": len(episode_data['images']),
            "action_shape": list(episode_data['actions'].shape),
            "created_at": datetime.now().isoformat()
        }
        
        metadata_path = episode_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def generate_distance_aware_dataset(self):
        """거리별 특화 증강 데이터셋 생성"""
        logger.info("🎯 거리별 특화 증강 시작!")
        logger.info("=" * 60)
        
        # 원본 데이터 로드
        episodes = self.load_original_dataset()
        if not episodes:
            logger.error("❌ 로드할 에피소드가 없습니다.")
            return
        
        # 거리별 통계
        distance_stats = {}
        for episode in episodes:
            distance = episode['distance']
            if distance not in distance_stats:
                distance_stats[distance] = 0
            distance_stats[distance] += 1
        
        logger.info("📊 원본 데이터 분포:")
        for distance, count in distance_stats.items():
            logger.info(f"   {distance}: {count}개")
        
        # 거리별 증강 진행
        all_augmented_episodes = []
        episode_counter = 0
        
        for episode in episodes:
            distance = episode['distance']
            augmentation_factor = self.distance_augmentation_factors[distance]
            
            logger.info(f"📏 {episode['episode_id']} ({distance}) 증강 중... (배수: {augmentation_factor}x)")
            
            # 원본 에피소드 저장
            original_dir = self.OUTPUT_DIR / f"episode_{episode_counter:04d}"
            self.save_augmented_episode(episode, original_dir)
            episode_counter += 1
            
            # 증강된 에피소드 생성
            augmented_episodes = self.apply_distance_specific_augmentation(episode, distance)
            
            for aug_episode in augmented_episodes:
                aug_dir = self.OUTPUT_DIR / f"episode_{episode_counter:04d}"
                self.save_augmented_episode(aug_episode, aug_dir)
                all_augmented_episodes.append(aug_episode)
                episode_counter += 1
        
        # 최종 통계
        final_stats = {}
        for episode in all_augmented_episodes:
            distance = episode['distance']
            if distance not in final_stats:
                final_stats[distance] = 0
            final_stats[distance] += 1
        
        # 원본도 포함
        for episode in episodes:
            distance = episode['distance']
            final_stats[distance] += 1
        
        logger.info("🎉 거리별 특화 증강 완료!")
        logger.info("📊 최종 데이터 분포:")
        for distance, count in final_stats.items():
            original_count = distance_stats.get(distance, 0)
            augmentation_factor = self.distance_augmentation_factors[distance]
            logger.info(f"   {distance}: {count}개 (원본: {original_count}개, 배수: {augmentation_factor}x)")
        
        # 전체 통계 저장
        total_stats = {
            "total_episodes": len(episodes) + len(all_augmented_episodes),
            "original_episodes": len(episodes),
            "augmented_episodes": len(all_augmented_episodes),
            "distance_distribution": final_stats,
            "augmentation_factors": self.distance_augmentation_factors,
            "created_at": datetime.now().isoformat()
        }
        
        stats_path = self.OUTPUT_DIR / "augmentation_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(total_stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 증강 통계 저장: {stats_path}")
        logger.info(f"📁 증강된 데이터셋: {self.OUTPUT_DIR}")

def main():
    """메인 함수"""
    augmenter = DistanceAwareAugmentation()
    augmenter.generate_distance_aware_dataset()

if __name__ == "__main__":
    main()
