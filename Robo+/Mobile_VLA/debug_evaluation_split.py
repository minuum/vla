"""
🔍 평가 분할 및 100% 정확도 문제 진단
학습셋과 평가셋이 제대로 분리되었는지 확인
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

def debug_data_split():
    """데이터 분할 디버깅"""
    
    print("🔍 데이터 분할 디버깅 시작")
    print("=" * 50)
    
    # 원본 데이터 확인
    data_path = '../../ROS_action/mobile_vla_dataset'
    
    if os.path.isdir(data_path):
        h5_files = list(Path(data_path).glob("*.h5"))
    else:
        h5_files = [data_path]
    
    print(f"📊 원본 H5 파일 수: {len(h5_files)}")
    
    # 각 H5 파일의 내용 확인
    all_episodes = []
    for h5_file in h5_files:
        try:
            with h5py.File(h5_file, 'r') as f:
                if 'images' in f and 'actions' in f:
                    images = f['images'][:]  # [18, H, W, 3]
                    actions = f['actions'][:]  # [18, 3]
                    
                    # 첫 프레임만 사용
                    single_image = images[0]  # [H, W, 3]
                    single_action = actions[0]  # [3]
                    
                    all_episodes.append({
                        'file': h5_file.name,
                        'image': single_image,
                        'action': single_action,
                        'episode_id': f"{h5_file.stem}"
                    })
        except Exception as e:
            print(f"❌ {h5_file} 로드 실패: {e}")
    
    print(f"📊 총 에피소드 수: {len(all_episodes)}")
    
    # 데이터셋 생성
    processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
    full_dataset = FixedRoboVLMStyleDataset(data_path, processor, 'full')
    
    print(f"📊 데이터셋 크기: {len(full_dataset)}")
    
    # 분할 확인
    train_size = int(len(full_dataset) * 0.8)
    val_size = len(full_dataset) - train_size
    
    print(f"📊 분할 계획:")
    print(f"   - 훈련: {train_size}개")
    print(f"   - 검증: {val_size}개")
    print(f"   - 총합: {train_size + val_size}개")
    
    # 실제 분할
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size]
    )
    
    print(f"📊 실제 분할:")
    print(f"   - 훈련: {len(train_dataset)}개")
    print(f"   - 검증: {len(val_dataset)}개")
    
    # 검증 데이터의 인덱스 확인
    val_indices = val_dataset.indices
    print(f"📊 검증 데이터 인덱스: {val_indices[:10]}... (총 {len(val_indices)}개)")
    
    return train_dataset, val_dataset, full_dataset

def debug_model_performance():
    """모델 성능 디버깅"""
    
    print("\n🔍 모델 성능 디버깅")
    print("=" * 50)
    
    # 모델 로드
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = 'fixed_robovlms_model_best.pth'
    
    if not os.path.exists(model_path):
        print(f"❌ 모델 파일 없음: {model_path}")
        return
    
    processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
    model = FixedRoboVLMStyleSingleImageModel(
        processor=processor,
        dropout=0.2,
        use_claw_matrix=True,
        use_hierarchical=True,
        use_advanced_attention=True,
        z_axis_weight=0.05
    ).to(device)
    
    # 모델 로드
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
    
    # 데이터 로더 생성
    data_path = '../../ROS_action/mobile_vla_dataset'
    train_dataset, val_dataset, full_dataset = debug_data_split()
    
    # 검증 데이터만 사용
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )
    
    # 상세 성능 평가
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    predictions = []
    targets = []
    
    print(f"\n🎯 검증 데이터 상세 평가:")
    print(f"   - 검증 배치 수: {len(val_loader)}")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
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
                
                # 예측과 타겟 저장
                predictions.append(predicted_actions.cpu().numpy())
                targets.append(actions.cpu().numpy())
                
                print(f"   - 배치 {batch_idx}: Loss={loss.item():.6f}, MAE={mae.item():.6f}")
                
            except Exception as e:
                print(f"❌ 배치 {batch_idx} 평가 중 오류: {e}")
                continue
    
    # 평균 계산
    avg_loss = total_loss / len(val_loader)
    avg_mae = total_mae / len(val_loader)
    
    # 예측 정확도 계산
    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)
    
    print(f"\n📊 최종 성능:")
    print(f"   - 평균 손실: {avg_loss:.6f}")
    print(f"   - 평균 MAE: {avg_mae:.6f}")
    
    # 다양한 임계값으로 정확도 계산
    thresholds = [0.01, 0.05, 0.1, 0.2, 0.5]
    for threshold in thresholds:
        within_threshold = np.abs(predictions - targets) < threshold
        accuracy = np.mean(within_threshold) * 100
        print(f"   - 임계값 {threshold}: {accuracy:.2f}%")
    
    # 축별 분석
    axis_names = ['X축', 'Y축', 'Z축']
    for i, axis_name in enumerate(axis_names):
        axis_mae = np.mean(np.abs(predictions[:, i] - targets[:, i]))
        axis_rmse = np.sqrt(np.mean((predictions[:, i] - targets[:, i]) ** 2))
        print(f"   - {axis_name} MAE: {axis_mae:.6f}, RMSE: {axis_rmse:.6f}")
    
    # 예측값과 타겟값 비교
    print(f"\n🔍 예측값 vs 타겟값 샘플:")
    for i in range(min(5, len(predictions))):
        print(f"   샘플 {i}:")
        print(f"     예측: {predictions[i]}")
        print(f"     타겟: {targets[i]}")
        print(f"     차이: {np.abs(predictions[i] - targets[i])}")

def check_overfitting():
    """과적합 확인"""
    
    print("\n🔍 과적합 확인")
    print("=" * 50)
    
    # 훈련 데이터와 검증 데이터의 성능 비교
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 모델 로드
    processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
    model = FixedRoboVLMStyleSingleImageModel(
        processor=processor,
        dropout=0.2,
        use_claw_matrix=True,
        use_hierarchical=True,
        use_advanced_attention=True,
        z_axis_weight=0.05
    ).to(device)
    
    model_path = 'fixed_robovlms_model_best.pth'
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
    
    # 데이터 분할
    data_path = '../../ROS_action/mobile_vla_dataset'
    train_dataset, val_dataset, full_dataset = debug_data_split()
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    # 훈련 데이터 성능
    model.eval()
    train_loss = 0.0
    train_mae = 0.0
    
    with torch.no_grad():
        for batch in train_loader:
            images = batch['image'].float().to(device)
            actions = batch['action'].float().to(device)
            
            predicted_actions = model(images, "Navigate to target")
            
            z_weight = torch.tensor([1.0, 1.0, 0.05]).to(device)
            weighted_target = actions * z_weight.unsqueeze(0)
            weighted_pred = predicted_actions * z_weight.unsqueeze(0)
            
            loss = torch.nn.functional.mse_loss(weighted_pred, weighted_target)
            mae = torch.mean(torch.abs(predicted_actions - actions))
            
            train_loss += loss.item()
            train_mae += mae.item()
    
    # 검증 데이터 성능
    val_loss = 0.0
    val_mae = 0.0
    
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].float().to(device)
            actions = batch['action'].float().to(device)
            
            predicted_actions = model(images, "Navigate to target")
            
            z_weight = torch.tensor([1.0, 1.0, 0.05]).to(device)
            weighted_target = actions * z_weight.unsqueeze(0)
            weighted_pred = predicted_actions * z_weight.unsqueeze(0)
            
            loss = torch.nn.functional.mse_loss(weighted_pred, weighted_target)
            mae = torch.mean(torch.abs(predicted_actions - actions))
            
            val_loss += loss.item()
            val_mae += mae.item()
    
    avg_train_loss = train_loss / len(train_loader)
    avg_train_mae = train_mae / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    avg_val_mae = val_mae / len(val_loader)
    
    print(f"📊 훈련 vs 검증 성능:")
    print(f"   - 훈련 손실: {avg_train_loss:.6f}")
    print(f"   - 검증 손실: {avg_val_loss:.6f}")
    print(f"   - 훈련 MAE: {avg_train_mae:.6f}")
    print(f"   - 검증 MAE: {avg_val_mae:.6f}")
    
    # 과적합 판단
    loss_gap = abs(avg_train_loss - avg_val_loss)
    mae_gap = abs(avg_train_mae - avg_val_mae)
    
    print(f"\n🔍 과적합 분석:")
    print(f"   - 손실 차이: {loss_gap:.6f}")
    print(f"   - MAE 차이: {mae_gap:.6f}")
    
    if loss_gap > 0.01 or mae_gap > 0.01:
        print(f"   ⚠️ 과적합 의심: 훈련과 검증 성능 차이가 큼")
    else:
        print(f"   ✅ 과적합 없음: 훈련과 검증 성능이 비슷함")

if __name__ == "__main__":
    debug_data_split()
    debug_model_performance()
    check_overfitting()
