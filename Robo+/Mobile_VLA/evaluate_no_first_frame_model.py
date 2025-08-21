"""
🔍 첫 프레임 제외 모델 성능 평가
첫 프레임을 제외하고 훈련된 모델의 실제 성능을 평가
"""

import torch
import numpy as np
import h5py
import os
from pathlib import Path
from transformers import AutoProcessor
from torch.utils.data import DataLoader, Dataset, random_split

from fixed_robovlms_model import FixedRoboVLMStyleSingleImageModel

class NoFirstFrameEvaluationDataset(Dataset):
    """첫 프레임 제외 평가용 데이터셋"""
    
    def __init__(self, data_path, processor, split='train', frame_selection='random'):
        self.data_path = data_path
        self.processor = processor
        self.split = split
        self.frame_selection = frame_selection
        
        # H5 파일들 로드
        self.episodes = []
        self._load_episodes()
        
        print(f"📊 {split} 평가 데이터셋 로드 완료: {len(self.episodes)}개 에피소드")
        print(f"   - 프레임 선택: {frame_selection}")
        print(f"   - 첫 프레임 제외: True")
    
    def _load_episodes(self):
        """에피소드 로드 (첫 프레임 제외)"""
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
                        
                        # 첫 프레임 제외 (프레임 1-17만 사용)
                        valid_frames = list(range(1, 18))  # 1, 2, 3, ..., 17
                        
                        if self.frame_selection == 'random':
                            # 랜덤하게 프레임 선택
                            frame_idx = np.random.choice(valid_frames)
                        elif self.frame_selection == 'middle':
                            # 중간 프레임 선택
                            frame_idx = valid_frames[len(valid_frames)//2]  # 9
                        elif self.frame_selection == 'all':
                            # 모든 유효한 프레임을 개별 에피소드로 생성
                            for frame_idx in valid_frames:
                                single_image = images[frame_idx]  # [H, W, 3]
                                single_action = actions[frame_idx]  # [3]
                                
                                self.episodes.append({
                                    'image': single_image,
                                    'action': single_action,
                                    'episode_id': f"{h5_file.stem}_frame_{frame_idx}",
                                    'frame_idx': frame_idx,
                                    'original_file': h5_file.name
                                })
                            continue
                        
                        # 단일 프레임 선택
                        single_image = images[frame_idx]  # [H, W, 3]
                        single_action = actions[frame_idx]  # [3]
                        
                        self.episodes.append({
                            'image': single_image,
                            'action': single_action,
                            'episode_id': f"{h5_file.stem}_frame_{frame_idx}",
                            'frame_idx': frame_idx,
                            'original_file': h5_file.name
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

def create_evaluation_loaders(data_path, processor, batch_size=4, train_split=0.8, frame_selection='random'):
    """평가용 데이터 로더 생성"""
    
    # 전체 데이터셋 로드
    full_dataset = NoFirstFrameEvaluationDataset(data_path, processor, 'full', frame_selection)
    
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
        shuffle=False,
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
    
    print(f"📊 평가용 데이터 로더 생성 완료:")
    print(f"   - 훈련: {len(train_dataset)}개 에피소드")
    print(f"   - 검증: {len(val_dataset)}개 에피소드")
    print(f"   - 배치 크기: {batch_size}")
    print(f"   - 프레임 선택: {frame_selection}")
    
    return train_loader, val_loader

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
    accuracy_02 = np.mean(np.abs(predictions - targets) < 0.2) * 100
    
    print(f"   📊 {description} 결과:")
    print(f"      - 평균 손실: {avg_loss:.6f}")
    print(f"      - 평균 MAE: {avg_mae:.6f}")
    print(f"      - 평균 RMSE: {avg_rmse:.6f}")
    print(f"      - 정확도(0.1): {accuracy_01:.2f}%")
    print(f"      - 정확도(0.05): {accuracy_005:.2f}%")
    print(f"      - 정확도(0.2): {accuracy_02:.2f}%")
    
    # 축별 분석
    axis_names = ['X축', 'Y축', 'Z축']
    for i, axis_name in enumerate(axis_names):
        axis_mae = np.mean(np.abs(predictions[:, i] - targets[:, i]))
        axis_rmse = np.sqrt(np.mean((predictions[:, i] - targets[:, i]) ** 2))
        print(f"      - {axis_name} MAE: {axis_mae:.6f}, RMSE: {axis_rmse:.6f}")
    
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
        'accuracy_0.2': accuracy_02,
        'predictions': predictions.tolist(),
        'targets': targets.tolist()
    }

def main():
    """메인 평가 함수"""
    
    print("🔍 첫 프레임 제외 모델 성능 평가 시작!")
    print("=" * 60)
    
    # 설정
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = 'no_first_frame_model_best.pth'
    
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
    print(f"   - 훈련 타입: {checkpoint['config']['training_type']}")
    
    # 데이터 로더 생성
    data_path = '../../ROS_action/mobile_vla_dataset'
    
    # 1. 랜덤 프레임 평가
    print(f"\n🎯 1. 랜덤 프레임 평가:")
    print("-" * 40)
    
    _, val_loader_random = create_evaluation_loaders(
        data_path, processor, batch_size=4, frame_selection='random'
    )
    
    random_results = evaluate_model_on_loader(model, val_loader_random, device, "랜덤 프레임")
    
    # 2. 중간 프레임 평가
    print(f"\n🎯 2. 중간 프레임 평가:")
    print("-" * 40)
    
    _, val_loader_middle = create_evaluation_loaders(
        data_path, processor, batch_size=4, frame_selection='middle'
    )
    
    middle_results = evaluate_model_on_loader(model, val_loader_middle, device, "중간 프레임")
    
    # 3. 성능 비교
    print(f"\n📊 성능 비교:")
    print("=" * 60)
    
    print(f"| 평가 방식 | MAE | RMSE | 정확도(0.1) | 정확도(0.05) | 정확도(0.2) |")
    print(f"|-----------|-----|------|-------------|-------------|-------------|")
    print(f"| 랜덤 프레임 | {random_results['mae']:.6f} | {random_results['rmse']:.6f} | {random_results['accuracy_0.1']:.2f}% | {random_results['accuracy_0.05']:.2f}% | {random_results['accuracy_0.2']:.2f}% |")
    print(f"| 중간 프레임 | {middle_results['mae']:.6f} | {middle_results['rmse']:.6f} | {middle_results['accuracy_0.1']:.2f}% | {middle_results['accuracy_0.05']:.2f}% | {middle_results['accuracy_0.2']:.2f}% |")
    
    # 4. 이전 모델과 비교
    print(f"\n📊 이전 모델과 비교:")
    print("=" * 60)
    
    previous_models = {
        "첫 프레임 포함 (이전)": {"mae": 0.576, "accuracy_0.1": 48.89, "accuracy_0.2": 48.89},
        "첫 프레임 제외 (현재)": {"mae": random_results['mae'], "accuracy_0.1": random_results['accuracy_0.1'], "accuracy_0.2": random_results['accuracy_0.2']}
    }
    
    print(f"| 모델 | MAE | 정확도(0.1) | 정확도(0.2) | 개선도 |")
    print(f"|------|-----|-------------|-------------|--------|")
    
    for model_name, metrics in previous_models.items():
        print(f"| {model_name} | {metrics['mae']:.6f} | {metrics['accuracy_0.1']:.2f}% | {metrics['accuracy_0.2']:.2f}% | - |")
    
    # 개선도 계산
    mae_improvement = float((0.576 - random_results['mae']) / 0.576 * 100)
    accuracy_improvement = float((random_results['accuracy_0.1'] - 48.89) / 48.89 * 100)
    
    print(f"| 개선도 | {mae_improvement:+.1f}% | {accuracy_improvement:+.1f}% | - | - |")
    
    # 5. 결론
    print(f"\n🎯 결론:")
    print("=" * 60)
    
    if random_results['accuracy_0.1'] > 70:
        print(f"✅ 첫 프레임 제외 훈련 성공: 실제 액션 예측 성능이 크게 향상됨")
        print(f"   - 정확도(0.1): {random_results['accuracy_0.1']:.2f}% (이전: 48.89%)")
        print(f"   - MAE: {random_results['mae']:.6f} (이전: 0.576)")
    elif random_results['accuracy_0.1'] > 50:
        print(f"⚠️ 첫 프레임 제외 훈련 부분적 성공: 성능이 어느 정도 향상됨")
        print(f"   - 정확도(0.1): {random_results['accuracy_0.1']:.2f}% (이전: 48.89%)")
    else:
        print(f"❌ 첫 프레임 제외 훈련 실패: 성능 향상이 미미함")
        print(f"   - 정확도(0.1): {random_results['accuracy_0.1']:.2f}% (이전: 48.89%)")
    
    # 6. 결과 저장
    results = {
        'model_type': 'Fixed_RoboVLMs_Style_Without_First_Frame',
        'evaluation_type': 'no_first_frame_evaluation',
        'random_frame_results': random_results,
        'middle_frame_results': middle_results,
        'improvement': {
            'mae_improvement_percent': mae_improvement,
            'accuracy_improvement_percent': accuracy_improvement,
            'previous_mae': 0.576,
            'previous_accuracy': 48.89
        },
        'conclusion': {
            'training_success': bool(random_results['accuracy_0.1'] > 70),
            'performance_level': 'good' if random_results['accuracy_0.1'] > 70 else 'needs_improvement'
        }
    }
    
    import json
    with open('no_first_frame_evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 평가 결과 저장 완료: no_first_frame_evaluation_results.json")

if __name__ == "__main__":
    main()
