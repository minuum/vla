import os
import sys
import argparse
import logging
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import h5py
from PIL import Image
import numpy as np
from transformers import AutoProcessor
from tqdm import tqdm
import matplotlib.pyplot as plt

# 새로운 모델 파일에서 import
from advanced_multimodal_model_v2 import AdvancedMultimodalModelV2, AdvancedMultimodalTrainerV2, create_advanced_multimodal_model_v2

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

class AdvancedDataset:
    """고급 데이터셋 - 다양한 증강 기법 적용"""
    
    def __init__(self, data_path, transform=None, use_augmentation=True):
        self.data_path = data_path
        self.transform = transform
        self.use_augmentation = use_augmentation
        self.episodes = []
        
        # 데이터 로드
        self._load_episodes()
        
        # 증강 기법들
        self.augmentations = [
            self._brightness_contrast_augmentation,
            self._gaussian_noise_augmentation,
            self._rotation_augmentation,
            self._crop_augmentation,
            self._blur_augmentation
        ]
        
    def _load_episodes(self):
        """에피소드 데이터 로드"""
        if os.path.isfile(self.data_path):
            # 단일 H5 파일
            try:
                with h5py.File(self.data_path, 'r') as f:
                    if 'images' in f and 'actions' in f:
                        self.episodes.append({
                            'file': self.data_path,
                            'num_frames': len(f['images'])
                        })
                        logger.info(f"✅ {self.data_path} 로드 완료 - {len(f['images'])} 프레임")
            except Exception as e:
                logger.error(f"❌ {self.data_path} 로드 오류: {e}")
        else:
            # 디렉토리
            for root, dirs, files in os.walk(self.data_path):
                for file in files:
                    if file.endswith('.h5'):
                        file_path = os.path.join(root, file)
                        try:
                            with h5py.File(file_path, 'r') as f:
                                if 'images' in f and 'actions' in f:
                                    self.episodes.append({
                                        'file': file_path,
                                        'num_frames': len(f['images'])
                                    })
                                    logger.info(f"✅ {file_path} 로드 완료 - {len(f['images'])} 프레임")
                        except Exception as e:
                            logger.error(f"❌ {file_path} 로드 오류: {e}")
        
        logger.info(f"총 {len(self.episodes)}개 에피소드 로드 완료")
    
    def _brightness_contrast_augmentation(self, image):
        """밝기/대비 증강"""
        import random
        brightness_factor = random.uniform(0.8, 1.2)
        contrast_factor = random.uniform(0.8, 1.2)
        
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness_factor)
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast_factor)
        
        return image
    
    def _gaussian_noise_augmentation(self, image):
        """가우시안 노이즈 증강"""
        import random
        import cv2
        
        # PIL을 numpy로 변환
        img_array = np.array(image)
        
        # 노이즈 추가
        noise = np.random.normal(0, random.uniform(5, 15), img_array.shape)
        noisy_img = img_array + noise
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
        
        # numpy를 PIL로 변환
        return Image.fromarray(noisy_img)
    
    def _rotation_augmentation(self, image):
        """회전 증강"""
        import random
        angle = random.uniform(-15, 15)
        return image.rotate(angle, fillcolor=(128, 128, 128))
    
    def _crop_augmentation(self, image):
        """크롭 증강"""
        import random
        
        width, height = image.size
        crop_ratio = random.uniform(0.8, 0.95)
        
        new_width = int(width * crop_ratio)
        new_height = int(height * crop_ratio)
        
        left = random.randint(0, width - new_width)
        top = random.randint(0, height - new_height)
        right = left + new_width
        bottom = top + new_height
        
        cropped = image.crop((left, top, right, bottom))
        return cropped.resize((width, height))
    
    def _blur_augmentation(self, image):
        """블러 증강"""
        import random
        import cv2
        
        # PIL을 numpy로 변환
        img_array = np.array(image)
        
        # 블러 적용
        kernel_size = random.choice([3, 5])
        blurred = cv2.GaussianBlur(img_array, (kernel_size, kernel_size), 0)
        
        # numpy를 PIL로 변환
        return Image.fromarray(blurred)
    
    def __len__(self):
        total_frames = 0
        for episode in self.episodes:
            total_frames += episode['num_frames']
        return total_frames
    
    def __getitem__(self, idx):
        # 에피소드와 프레임 인덱스 찾기
        current_idx = 0
        for episode in self.episodes:
            if current_idx + episode['num_frames'] > idx:
                frame_idx = idx - current_idx
                break
            current_idx += episode['num_frames']
        else:
            # 마지막 에피소드의 마지막 프레임
            episode = self.episodes[-1]
            frame_idx = episode['num_frames'] - 1
        
        # 데이터 로드
        with h5py.File(episode['file'], 'r') as f:
            # 이미지 로드
            image_data = f['images'][frame_idx]
            if len(image_data.shape) == 3:
                image = Image.fromarray(image_data)
            else:
                image = Image.fromarray(image_data, mode='RGB')
            
            # 액션 로드 (2D로 변환)
            action_data = f['actions'][frame_idx]
            if len(action_data) >= 3:
                action = torch.tensor([action_data[0], action_data[1]], dtype=torch.float32)  # linear_x, linear_y만
            else:
                action = torch.tensor([0.0, 0.0], dtype=torch.float32)
            
            # 텍스트 (에피소드 파일명에서 추출)
            episode_name = os.path.basename(episode['file'])
            text = f"Episode: {episode_name}, Frame: {frame_idx}"
        
        # 증강 적용
        if self.use_augmentation and np.random.random() < 0.3:
            augmentation = np.random.choice(self.augmentations)
            image = augmentation(image)
        
        # 변환 적용
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'action': action,
            'text': text,
            'episode_id': episode['file']
        }

def custom_collate_fn(batch):
    """PIL 이미지를 처리하는 커스텀 collate 함수"""
    images = [item['image'] for item in batch]
    actions = torch.stack([item['action'] for item in batch])
    texts = [item['text'] for item in batch]
    episode_ids = [item['episode_id'] for item in batch]
    
    return {
        'image': images,  # PIL 이미지 리스트
        'action': actions,
        'text': texts,
        'episode_id': episode_ids
    }

def create_advanced_data_loaders(data_path, batch_size=4, train_ratio=0.7, val_ratio=0.15):
    """고급 데이터 로더 생성"""
    # 데이터셋 생성
    dataset = AdvancedDataset(data_path, use_augmentation=True)
    
    # 데이터 분할
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # 데이터 로더 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    return train_loader, val_loader, test_loader

def evaluate_final_performance(model, test_loader, device, output_path):
    """최종 성능 평가"""
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_losses = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="평가 중"):
            images = batch['image']  # PIL 이미지 리스트
            actions = batch['action'].to(device)
            texts = batch['text']
            
            # 예측
            predicted_actions = model(images, texts)
            
            # 손실 계산
            loss = nn.MSELoss()(predicted_actions, actions)
            all_losses.append(loss.item())
            
            # 예측과 타겟 저장
            all_predictions.append(predicted_actions.cpu())
            all_targets.append(actions.cpu())
    
    # 결과 결합
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # 메트릭 계산
    mse = torch.mean((all_predictions - all_targets) ** 2).item()
    mae = torch.mean(torch.abs(all_predictions - all_targets)).item()
    rmse = torch.sqrt(torch.mean((all_predictions - all_targets) ** 2)).item()
    
    # R² 스코어
    ss_res = torch.sum((all_targets - all_predictions) ** 2)
    ss_tot = torch.sum((all_targets - torch.mean(all_targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot).item()
    
    # 정확도 계산 (다양한 임계값)
    thresholds = [0.1, 0.2, 0.5, 1.0]
    accuracies = {}
    
    for threshold in thresholds:
        within_threshold = torch.all(torch.abs(all_predictions - all_targets) < threshold, dim=1)
        accuracy = torch.mean(within_threshold.float()).item() * 100
        accuracies[f'acc_{threshold}'] = accuracy
    
    # 개별 축 정확도
    axis_accuracies = {}
    for i, axis_name in enumerate(['linear_x', 'linear_y']):
        for threshold in thresholds:
            within_threshold = torch.abs(all_predictions[:, i] - all_targets[:, i]) < threshold
            accuracy = torch.mean(within_threshold.float()).item() * 100
            axis_accuracies[f'{axis_name}_acc_{threshold}'] = accuracy
    
    # 결과 저장
    results = {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'accuracies': accuracies,
        'axis_accuracies': axis_accuracies,
        'num_samples': len(all_predictions)
    }
    
    with open(os.path.join(output_path, 'final_evaluation.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # 결과 출력
    logger.info("📊 최종 성능 평가 결과:")
    logger.info(f"  - MSE: {mse:.6f}")
    logger.info(f"  - MAE: {mae:.6f}")
    logger.info(f"  - RMSE: {rmse:.6f}")
    logger.info(f"  - R² Score: {r2:.4f}")
    
    for threshold, acc in accuracies.items():
        logger.info(f"  - 정확도 ({threshold}): {acc:.2f}%")
    
    for axis, acc in axis_accuracies.items():
        logger.info(f"  - {axis}: {acc:.2f}%")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Case 3: 고급 멀티모달 모델 훈련')
    parser.add_argument('--data_path', type=str, required=True, help='데이터 경로')
    parser.add_argument('--output_dir', type=str, default='case3_results', help='출력 디렉토리')
    parser.add_argument('--num_epochs', type=int, default=10, help='에포크 수')
    parser.add_argument('--batch_size', type=int, default=4, help='배치 크기')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='학습률')
    parser.add_argument('--patience', type=int, default=5, help='조기 종료 인내심')
    
    args = parser.parse_args()
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🚀 디바이스: {device}")
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Kosmos2 프로세서 로드
    processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
    logger.info("✅ Kosmos2 프로세서 로드 완료")
    
    # 데이터 로더 생성
    train_loader, val_loader, test_loader = create_advanced_data_loaders(
        args.data_path, args.batch_size
    )
    logger.info(f"✅ 데이터 로더 생성 완료 - 훈련: {len(train_loader)}, 검증: {len(val_loader)}, 테스트: {len(test_loader)}")
    
    # 모델 생성
    model, trainer = create_advanced_multimodal_model_v2(
        processor=processor,
        device=device,
        vision_dim=1024,
        language_dim=2048,  # Kosmos2 text 모델 출력 차원
        action_dim=2,
        hidden_dim=512,
        dropout=0.3,
        use_hierarchical_planning=True
    )
    logger.info("✅ 고급 멀티모달 모델 V2 생성 완료")
    
    # 훈련 루프
    best_mae = float('inf')
    patience_counter = 0
    training_history = []
    
    logger.info("🎯 Case 3 훈련 시작!")
    
    for epoch in range(args.num_epochs):
        # 훈련
        model.train()
        train_losses = []
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Train]")
        
        for batch in train_pbar:
            try:
                loss = trainer.train_step(batch)
                train_losses.append(loss)
                train_pbar.set_postfix({'loss': f'{loss:.4f}'})
            except Exception as e:
                logger.error(f"❌ 훈련 배치 오류: {e}")
                continue
        
        avg_train_loss = np.mean(train_losses)
        
        # 검증
        model.eval()
        val_losses = []
        val_maes = []
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Val]")
        
        for batch in val_pbar:
            try:
                loss, mae = trainer.validate_step(batch)
                val_losses.append(loss)
                val_maes.append(mae)
                val_pbar.set_postfix({'loss': f'{loss:.4f}', 'mae': f'{mae:.4f}'})
            except Exception as e:
                logger.error(f"❌ 검증 배치 오류: {e}")
                continue
        
        avg_val_loss = np.mean(val_losses)
        avg_val_mae = np.mean(val_maes)
        
        # 히스토리 저장
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_mae': avg_val_mae
        })
        
        logger.info(f"📊 Epoch {epoch+1}/{args.num_epochs} - "
                   f"Train Loss: {avg_train_loss:.4f}, "
                   f"Val Loss: {avg_val_loss:.4f}, "
                   f"Val MAE: {avg_val_mae:.4f}")
        
        # 모델 저장
        if avg_val_mae < best_mae:
            best_mae = avg_val_mae
            patience_counter = 0
            
            # 모델 저장
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'best_mae': best_mae,
                'training_history': training_history
            }, os.path.join(args.output_dir, 'best_model.pth'))
            
            logger.info(f"💾 모델 저장 (MAE: {best_mae:.4f})")
        else:
            patience_counter += 1
        
        # 조기 종료
        if patience_counter >= args.patience:
            logger.info(f"⏹️ 조기 종료 (patience: {args.patience})")
            break
    
    # 최종 성능 평가
    logger.info("🔍 최종 성능 평가 중...")
    final_results = evaluate_final_performance(model, test_loader, device, args.output_dir)
    
    # 훈련 히스토리 저장
    with open(os.path.join(args.output_dir, 'training_history.json'), 'w') as f:
        json.dump(training_history, f, indent=2)
    
    # 목표 달성 여부 확인
    target_mae = 0.5
    achieved = best_mae < target_mae
    
    logger.info(f"✅ Case 3 훈련 완료! - 최고 MAE: {best_mae:.6f} - "
               f"목표 달성: {'✅' if achieved else '❌'} (목표: < {target_mae}) - "
               f"최종 에포크: {epoch+1} - "
               f"결과 저장: {args.output_dir}")

if __name__ == '__main__':
    main()
