"""
🔍 실제 성능 평가 (첫 프레임 0 고정 고려)
첫 프레임이 0인 것이 고정이라는 점을 반영한 정확한 성능 평가
"""

import torch
import numpy as np
import h5py
import os
from pathlib import Path
from transformers import AutoProcessor
from torch.utils.data import DataLoader, Dataset, random_split

from fixed_robovlms_model import FixedRoboVLMStyleSingleImageModel
from train_fixed_robovlms import FixedRoboVLMStyleDataset

class RealisticRoboVLMStyleDataset(Dataset):
    """현실적인 RoboVLMs 스타일 데이터셋 (첫 프레임 0 고정 고려)"""
    
    def __init__(self, data_path, processor, split='train', use_middle_frame=True):
        self.data_path = data_path
        self.processor = processor
        self.split = split
        self.use_middle_frame = use_middle_frame
        
        # H5 파일들 로드
        self.episodes = []
        self._load_episodes()
        
        print(f"📊 {split} 데이터셋 로드 완료: {len(self.episodes)}개 에피소드")
        print(f"   - 중간 프레임 사용: {use_middle_frame}")
    
    def _load_episodes(self):
        """에피소드 로드 (중간 프레임 사용)"""
        if os.path.isdir(self.data_path):
            h5_files = list(Path(self.data_path).glob("*.h5"))
        else:
            h5_files = [self.data_path]
        
        for h5_file in h5_files:
            try:
                with h5py.File(h5_file, 'r') as f:
                    if 'images' in f and 'actions' in f:
                        images = f['images'][:]  # [18, H, W, 3]
                        actions = f['actions'][:]  # [18, 3]
                        
                        if self.use_middle_frame:
                            # 중간 프레임 사용 (프레임 9, 18개 중 10번째)
                            frame_idx = 9
                            single_image = images[frame_idx]  # [H, W, 3]
                            single_action = actions[frame_idx]  # [3]
                        else:
                            # 첫 프레임 사용 (0 고정)
                            single_image = images[0]  # [H, W, 3]
                            single_action = actions[0]  # [3]
                        
                        self.episodes.append({
                            'image': single_image,  # [H, W, 3]
                            'action': single_action,  # [3]
                            'episode_id': f"{h5_file.stem}",
                            'frame_idx': frame_idx if self.use_middle_frame else 0
                        })
            except Exception as e:
                print(f"❌ {h5_file} 로드 실패: {e}")
    
    def __len__(self):
        return len(self.episodes)
    
    def __getitem__(self, idx):
        episode = self.episodes[idx]
        
        # 이미지: [H, W, 3] → [3, H, W] (PyTorch 형식)
        image = episode['image']  # [H, W, 3]
        image = np.transpose(image, (2, 0, 1))  # [3, H, W]
        
        # 0-1 범위로 정규화
        if image.max() > 1:
            image = image.astype(np.float32) / 255.0
        
        # 액션: [3]
        action = episode['action']  # [3]
        
        return {
            'image': image,  # [3, H, W]
            'action': action,  # [3]
            'episode_id': episode['episode_id'],
            'frame_idx': episode['frame_idx']
        }

def create_realistic_data_loaders(data_path, processor, batch_size=4, train_split=0.8, use_middle_frame=True):
    """현실적인 데이터 로더 생성"""
    
    # 전체 데이터셋 로드
    full_dataset = RealisticRoboVLMStyleDataset(data_path, processor, 'full', use_middle_frame)
    
    # 훈련/검증 분할
    total_size = len(full_dataset)
    train_size = int(total_size * train_split)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size]
    )
    
    # 데이터 로더 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )
    
    print(f"📊 현실적인 데이터 로더 생성 완료:")
    print(f"   - 훈련: {len(train_dataset)}개 에피소드")
    print(f"   - 검증: {len(val_dataset)}개 에피소드")
    print(f"   - 배치 크기: {batch_size}")
    print(f"   - 중간 프레임 사용: {use_middle_frame}")
    
    return train_loader, val_loader

def evaluate_realistic_performance():
    """현실적인 성능 평가"""
    
    print("🔍 현실적인 성능 평가 시작")
    print("=" * 60)
    
    # 설정
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = 'fixed_robovlms_model_best.pth'
    
    if not os.path.exists(model_path):
        print(f"❌ 모델 파일 없음: {model_path}")
        return
    
    # 프로세서 로드
    processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
    
    # 모델 로드
    model = FixedRoboVLMStyleSingleImageModel(
        processor=processor,
        dropout=0.2,
        use_claw_matrix=True,
        use_hierarchical=True,
        use_advanced_attention=True,
        z_axis_weight=0.05
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # 동적 어댑터 초기화
    dummy_image = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        try:
            _ = model(dummy_image, "Navigate to target")
        except:
            pass
    
    # 호환되는 키만 로드
    model_dict = model.state_dict()
    checkpoint_dict = checkpoint['model_state_dict']
    compatible_dict = {k: v for k, v in checkpoint_dict.items() 
                      if k in model_dict and model_dict[k].shape == v.shape}
    model_dict.update(compatible_dict)
    model.load_state_dict(model_dict)
    
    print(f"✅ 모델 로드 완료")
    print(f"   - 에포크: {checkpoint['epoch']}")
    print(f"   - 훈련 손실: {checkpoint['train_loss']:.6f}")
    print(f"   - 검증 손실: {checkpoint['val_loss']:.6f}")
    
    # 1. 첫 프레임 평가 (0 고정)
    print(f"\n🎯 1. 첫 프레임 평가 (0 고정):")
    print("-" * 40)
    
    data_path = '../../ROS_action/mobile_vla_dataset'
    _, val_loader_first = create_realistic_data_loaders(
        data_path, processor, batch_size=4, use_middle_frame=False
    )
    
    model.eval()
    first_frame_results = evaluate_model_on_loader(model, val_loader_first, device, "첫 프레임")
    
    # 2. 중간 프레임 평가 (실제 액션)
    print(f"\n🎯 2. 중간 프레임 평가 (실제 액션):")
    print("-" * 40)
    
    _, val_loader_middle = create_realistic_data_loaders(
        data_path, processor, batch_size=4, use_middle_frame=True
    )
    
    middle_frame_results = evaluate_model_on_loader(model, val_loader_middle, device, "중간 프레임")
    
    # 3. 성능 비교
    print(f"\n📊 성능 비교:")
    print("=" * 60)
    
    print(f"| 평가 방식 | MAE | RMSE | 정확도(0.1) | 정확도(0.05) |")
    print(f"|-----------|-----|------|-------------|-------------|")
    print(f"| 첫 프레임 | {first_frame_results['mae']:.6f} | {first_frame_results['rmse']:.6f} | {first_frame_results['accuracy_0.1']:.2f}% | {first_frame_results['accuracy_0.05']:.2f}% |")
    print(f"| 중간 프레임 | {middle_frame_results['mae']:.6f} | {middle_frame_results['rmse']:.6f} | {middle_frame_results['accuracy_0.1']:.2f}% | {middle_frame_results['accuracy_0.05']:.2f}% |")
    
    # 4. 결론
    print(f"\n🎯 결론:")
    print("=" * 60)
    
    if first_frame_results['accuracy_0.1'] > 95:
        print(f"✅ 첫 프레임 평가: 모델이 0을 잘 예측함 (정확도: {first_frame_results['accuracy_0.1']:.2f}%)")
    else:
        print(f"⚠️ 첫 프레임 평가: 모델이 0을 제대로 예측하지 못함 (정확도: {first_frame_results['accuracy_0.1']:.2f}%)")
    
    if middle_frame_results['accuracy_0.1'] > 70:
        print(f"✅ 중간 프레임 평가: 모델이 실제 액션을 잘 예측함 (정확도: {middle_frame_results['accuracy_0.1']:.2f}%)")
    elif middle_frame_results['accuracy_0.1'] > 50:
        print(f"⚠️ 중간 프레임 평가: 모델이 실제 액션을 어느 정도 예측함 (정확도: {middle_frame_results['accuracy_0.1']:.2f}%)")
    else:
        print(f"❌ 중간 프레임 평가: 모델이 실제 액션을 제대로 예측하지 못함 (정확도: {middle_frame_results['accuracy_0.1']:.2f}%)")
    
    # 5. 결과 저장
    results = {
        'model_type': 'Fixed_RoboVLMs_Style_Single_Image',
        'evaluation_type': 'realistic_with_first_frame_zero',
        'first_frame_results': first_frame_results,
        'middle_frame_results': middle_frame_results,
        'conclusion': {
            'first_frame_zero_accuracy': first_frame_results['accuracy_0.1'],
            'middle_frame_real_accuracy': middle_frame_results['accuracy_0.1'],
            'model_performance': 'good' if middle_frame_results['accuracy_0.1'] > 70 else 'needs_improvement'
        }
    }
    
    import json
    with open('realistic_evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 결과 저장 완료: realistic_evaluation_results.json")

def evaluate_model_on_loader(model, data_loader, device, description):
    """데이터 로더에서 모델 성능 평가"""
    
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_rmse = 0.0
    predictions = []
    targets = []
    
    print(f"   📊 {description} 평가 중...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            try:
                images = batch['image'].float().to(device)
                actions = batch['action'].float().to(device)
                
                # 예측
                predicted_actions = model(images, "Navigate to target")
                
                # 손실 계산
                z_weight = torch.tensor([1.0, 1.0, 0.05]).to(device)
                weighted_target = actions * z_weight.unsqueeze(0)
                weighted_pred = predicted_actions * z_weight.unsqueeze(0)
                
                loss = torch.nn.functional.mse_loss(weighted_pred, weighted_target)
                total_loss += loss.item()
                
                # MAE 계산
                mae = torch.mean(torch.abs(predicted_actions - actions))
                total_mae += mae.item()
                
                # RMSE 계산
                rmse = torch.sqrt(torch.mean((predicted_actions - actions) ** 2))
                total_rmse += rmse.item()
                
                # 예측과 타겟 저장
                predictions.append(predicted_actions.cpu().numpy())
                targets.append(actions.cpu().numpy())
                
            except Exception as e:
                print(f"❌ 배치 {batch_idx} 평가 중 오류: {e}")
                continue
    
    # 평균 계산
    avg_loss = total_loss / len(data_loader)
    avg_mae = total_mae / len(data_loader)
    avg_rmse = total_rmse / len(data_loader)
    
    # 예측 정확도 계산
    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)
    
    # 다양한 임계값으로 정확도 계산
    accuracy_01 = np.mean(np.abs(predictions - targets) < 0.1) * 100
    accuracy_005 = np.mean(np.abs(predictions - targets) < 0.05) * 100
    
    print(f"   📊 {description} 결과:")
    print(f"      - 평균 손실: {avg_loss:.6f}")
    print(f"      - 평균 MAE: {avg_mae:.6f}")
    print(f"      - 평균 RMSE: {avg_rmse:.6f}")
    print(f"      - 정확도(0.1): {accuracy_01:.2f}%")
    print(f"      - 정확도(0.05): {accuracy_005:.2f}%")
    
    # 샘플 출력
    print(f"   🔍 {description} 샘플:")
    for i in range(min(3, len(predictions))):
        print(f"      샘플 {i}: 예측={predictions[i]}, 타겟={targets[i]}")
    
    return {
        'loss': avg_loss,
        'mae': avg_mae,
        'rmse': avg_rmse,
        'accuracy_0.1': accuracy_01,
        'accuracy_0.05': accuracy_005,
        'predictions': predictions.tolist(),
        'targets': targets.tolist()
    }

if __name__ == "__main__":
    evaluate_realistic_performance()
