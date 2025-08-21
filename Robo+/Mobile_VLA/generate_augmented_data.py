#!/usr/bin/env python3
"""
🎯 증강된 데이터 미리 생성 및 저장
"""
import sys
from pathlib import Path
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import json
import pickle
import random
from datetime import datetime
import logging

# 프로젝트 경로 설정
ROOT_DIR = Path("/home/billy/25-1kp/vla/Robo+/Mobile_VLA")
DATA_DIR = Path("/home/billy/25-1kp/vla/ROS_action/mobile_vla_dataset")

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AugmentedDataGenerator:
    """증강된 데이터를 미리 생성하는 클래스"""
    
    def __init__(self, augmentation_factor=10):
        self.augmentation_factor = augmentation_factor
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 증강 설정
        self.augmentation_config = {
            'horizontal_flip_prob': 0.5,
            'forward_backward_flip_prob': 0.3,
            'action_noise_prob': 0.8,
            'speed_variation_prob': 0.3,
            'start_stop_pattern_prob': 0.2,
            'action_noise_std': 0.005,
            'speed_scale_range': (0.8, 1.2)
        }
        
        logger.info(f"🎯 증강 데이터 생성기 초기화 (배수: {augmentation_factor}x)")
    
    def load_original_dataset(self):
        """원본 데이터셋 로드"""
        logger.info("📂 원본 데이터셋 로드 중...")
        
        episodes = []
        
        # .h5 파일들 처리
        h5_files = list(DATA_DIR.glob("*.h5"))
        logger.info(f"📁 발견된 .h5 파일: {len(h5_files)}개")
        
        for h5_file in h5_files:
            try:
                import h5py
                with h5py.File(h5_file, 'r') as f:
                    episode_data = {
                        'images': [],
                        'actions': [],
                        'episode_id': h5_file.stem
                    }
                    
                    # 이미지 데이터 로드
                    if 'images' in f:
                        images_data = f['images'][:]
                        for img_data in images_data:
                            # numpy 배열을 PIL Image로 변환
                            if img_data.dtype == np.uint8:
                                image = Image.fromarray(img_data)
                            else:
                                # 정규화된 데이터인 경우 0-255로 변환
                                img_data = (img_data * 255).astype(np.uint8)
                                image = Image.fromarray(img_data)
                            episode_data['images'].append(image)
                    
                    # 액션 데이터 로드
                    if 'actions' in f:
                        actions = f['actions'][:]
                        episode_data['actions'] = actions
                    
                    if len(episode_data['images']) > 0 and len(episode_data['actions']) > 0:
                        episodes.append(episode_data)
                        logger.info(f"✅ {h5_file.name} 로드 완료: {len(episode_data['images'])} 프레임, {len(episode_data['actions'])} 액션")
                        
            except Exception as e:
                logger.warning(f"⚠️ {h5_file.name} 로드 실패: {e}")
                continue
        
        # 디렉토리 형태의 에피소드들도 처리
        for episode_dir in sorted(DATA_DIR.glob("episode_*")):
            if episode_dir.is_dir():
                episode_data = {
                    'images': [],
                    'actions': [],
                    'episode_id': episode_dir.name
                }
                
                # 이미지 로드
                image_files = sorted(episode_dir.glob("*.jpg"))
                for img_file in image_files:
                    image = Image.open(img_file).convert('RGB')
                    episode_data['images'].append(image)
                
                # 액션 로드
                action_file = episode_dir / "actions.npy"
                if action_file.exists():
                    actions = np.load(action_file)
                    episode_data['actions'] = actions
                
                if len(episode_data['images']) > 0 and len(episode_data['actions']) > 0:
                    episodes.append(episode_data)
        
        logger.info(f"✅ 원본 데이터셋 로드 완료: {len(episodes)} 에피소드")
        return episodes
    
    def augment_episode(self, episode_data):
        """단일 에피소드 증강"""
        augmented_episodes = [episode_data]  # 원본 포함
        
        for i in range(self.augmentation_factor - 1):
            aug_episode = self._apply_augmentations(episode_data.copy())
            augmented_episodes.append(aug_episode)
        
        return augmented_episodes
    
    def _apply_augmentations(self, episode_data):
        """증강 적용"""
        images = episode_data['images'].copy()
        actions = episode_data['actions'].copy()
        
        # 1. 좌우 대칭 (50% 확률)
        if random.random() < self.augmentation_config['horizontal_flip_prob']:
            images = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in images]
            actions[:, 1] = -actions[:, 1]  # Y축 부호 변경
        
        # 2. 전진/후진 뒤집기 (30% 확률)
        if random.random() < self.augmentation_config['forward_backward_flip_prob']:
            images = images[::-1]  # 시간축 뒤집기
            actions = actions[::-1]  # 액션도 뒤집기
            actions[:, 0] = -actions[:, 0]  # X축 부호 변경
        
        # 3. 액션 노이즈 (80% 확률)
        if random.random() < self.augmentation_config['action_noise_prob']:
            # X축 노이즈
            x_noise = np.random.normal(0, self.augmentation_config['action_noise_std'], actions[:, 0].shape)
            actions[:, 0] += x_noise
            
            # Y축 노이즈 (더 작게)
            y_noise = np.random.normal(0, self.augmentation_config['action_noise_std'] * 0.5, actions[:, 1].shape)
            actions[:, 1] += y_noise
        
        # 4. 속도 변화 (30% 확률)
        if random.random() < self.augmentation_config['speed_variation_prob']:
            speed_scale = random.uniform(*self.augmentation_config['speed_scale_range'])
            actions[:, 0] *= speed_scale  # X축만 스케일링
        
        # 5. 시작-정지 패턴 (20% 확률)
        if random.random() < self.augmentation_config['start_stop_pattern_prob']:
            if random.random() < 0.5:
                # 시작 부분 정지
                stop_frames = random.randint(1, 3)
                actions[:stop_frames, :] = 0
            else:
                # 중간 부분 정지
                mid_point = len(actions) // 2
                actions[mid_point:mid_point+1, :] = 0
        
        # 범위 제한
        actions = np.clip(actions, -1.15, 1.15)
        
        # 증강된 에피소드 정보 업데이트
        episode_data['images'] = images
        episode_data['actions'] = actions
        episode_data['augmentation_type'] = self._get_augmentation_type()
        
        return episode_data
    
    def _get_augmentation_type(self):
        """적용된 증강 타입 반환"""
        types = []
        if random.random() < self.augmentation_config['horizontal_flip_prob']:
            types.append('horizontal_flip')
        if random.random() < self.augmentation_config['forward_backward_flip_prob']:
            types.append('forward_backward_flip')
        if random.random() < self.augmentation_config['action_noise_prob']:
            types.append('action_noise')
        if random.random() < self.augmentation_config['speed_variation_prob']:
            types.append('speed_variation')
        if random.random() < self.augmentation_config['start_stop_pattern_prob']:
            types.append('start_stop_pattern')
        
        return types if types else ['original']
    
    def generate_augmented_dataset(self):
        """전체 증강 데이터셋 생성"""
        logger.info("🎯 증강 데이터셋 생성 시작...")
        
        # 원본 데이터 로드
        original_episodes = self.load_original_dataset()
        
        # 증강된 데이터 생성
        all_episodes = []
        augmentation_stats = {
            'horizontal_flip': 0,
            'forward_backward_flip': 0,
            'action_noise': 0,
            'speed_variation': 0,
            'start_stop_pattern': 0,
            'original': 0
        }
        
        for i, episode in enumerate(original_episodes):
            logger.info(f"📊 에피소드 {i+1}/{len(original_episodes)} 증강 중...")
            
            augmented_episodes = self.augment_episode(episode)
            
            for aug_episode in augmented_episodes:
                all_episodes.append(aug_episode)
                
                # 통계 업데이트
                aug_type = aug_episode.get('augmentation_type', ['original'])
                for aug in aug_type:
                    if aug in augmentation_stats:
                        augmentation_stats[aug] += 1
        
        logger.info(f"✅ 증강 완료: {len(all_episodes)} 에피소드 생성")
        logger.info("📊 증강 통계:")
        for aug_type, count in augmentation_stats.items():
            logger.info(f"   {aug_type}: {count} 에피소드")
        
        return all_episodes, augmentation_stats
    
    def save_augmented_dataset(self, episodes, stats):
        """증강된 데이터셋 저장"""
        logger.info("💾 증강된 데이터셋 저장 중...")
        
        # 저장 디렉토리 생성
        save_dir = ROOT_DIR / "augmented_dataset"
        save_dir.mkdir(exist_ok=True)
        
        # 메타데이터 저장
        metadata = {
            'total_episodes': len(episodes),
            'original_episodes': len(episodes) // self.augmentation_factor,
            'augmentation_factor': self.augmentation_factor,
            'augmentation_stats': stats,
            'generation_date': datetime.now().isoformat(),
            'augmentation_config': self.augmentation_config
        }
        
        with open(save_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # 에피소드별 데이터 저장
        for i, episode in enumerate(episodes):
            episode_dir = save_dir / f"episode_{i:04d}"
            episode_dir.mkdir(exist_ok=True)
            
            # 이미지 저장
            for j, image in enumerate(episode['images']):
                image_path = episode_dir / f"frame_{j:02d}.jpg"
                image.save(image_path)
            
            # 액션 저장
            actions_path = episode_dir / "actions.npy"
            np.save(actions_path, episode['actions'])
            
            # 에피소드 메타데이터
            episode_meta = {
                'episode_id': episode['episode_id'],
                'augmentation_type': episode.get('augmentation_type', ['original']),
                'num_frames': len(episode['images']),
                'action_shape': episode['actions'].shape
            }
            
            with open(episode_dir / "metadata.json", 'w') as f:
                json.dump(episode_meta, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ 증강된 데이터셋 저장 완료: {save_dir}")
        return save_dir

def main():
    """메인 실행 함수"""
    print("🎯 증강된 데이터 생성 시작!")
    print("=" * 50)
    
    # 증강 데이터 생성기 초기화
    generator = AugmentedDataGenerator(augmentation_factor=10)
    
    # 증강된 데이터셋 생성
    episodes, stats = generator.generate_augmented_dataset()
    
    # 데이터셋 저장
    save_dir = generator.save_augmented_dataset(episodes, stats)
    
    print("\n🎉 증강된 데이터 생성 완료!")
    print(f"📁 저장 위치: {save_dir}")
    print(f"📊 총 에피소드: {len(episodes)}")
    print(f"📈 증강 배수: {generator.augmentation_factor}x")
    
    # 통계 출력
    print("\n📊 증강 통계:")
    for aug_type, count in stats.items():
        print(f"   {aug_type}: {count} 에피소드")

if __name__ == "__main__":
    main()
