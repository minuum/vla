"""
🎯 2D 액션 최적화 모델 평가 스크립트
훈련된 2D 액션 모델의 성능을 평가하고 기존 모델들과 비교
"""

import torch
import torch.nn as nn
import numpy as np
import h5py
import os
from pathlib import Path
from transformers import AutoProcessor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import json
from PIL import Image

from optimized_2d_action_model import Optimized2DActionModel, Optimized2DActionDataset

class Optimized2DEvaluationDataset(Dataset):
    """2D 액션 평가용 데이터셋"""
    
    def __init__(self, data_path, processor, split='val'):
        self.data_path = data_path
        self.processor = processor
        self.split = split
        
        # H5 파일들 로드
        self.episodes = []
        self._load_episodes()
        
        print(f"📊 {split} 2D 액션 평가 데이터셋 로드 완료: {len(self.episodes)}개 에피소드")
    
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
                        
                        # 모든 유효한 프레임을 평가에 사용
                        for frame_idx in valid_frames:
                            single_image = images[frame_idx]  # [H, W, 3]
                            single_action = actions[frame_idx]  # [3]
                            
                            # 2D 액션으로 변환 (Z축 제외)
                            action_2d = single_action[:2]  # [linear_x, linear_y]만 사용
                            
                            self.episodes.append({
                                'image': single_image,
                                'action': action_2d,  # 2D 액션
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
        
        # 2D 액션: [2]
        action = episode['action']  # [2]
        
        return {
            'image': image,  # [3, H, W]
            'action': action,  # [2]
            'episode_id': episode['episode_id']
        }

def evaluate_2d_model(model, data_loader, device):
    """2D 액션 모델 평가"""
    model.eval()
    
    total_loss = 0.0
    total_mae = 0.0
    total_rmse = 0.0
    total_samples = 0
    
    # 성공률 계산을 위한 임계값
    success_thresholds = [0.1, 0.05, 0.01]
    success_counts = {f"accuracy_{int(t*100)}": 0 for t in success_thresholds}
    
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="평가 중"):
            images = batch['image'].float().to(device)
            actions = batch['action'].float().to(device)
            
            # 2D 액션 예측
            predicted_actions = model(images, "Navigate to target")
            
            # 손실 계산
            loss = nn.functional.mse_loss(predicted_actions, actions)
            total_loss += loss.item()
            
            # MAE 계산
            mae = nn.functional.l1_loss(predicted_actions, actions)
            total_mae += mae.item()
            
            # RMSE 계산
            rmse = torch.sqrt(nn.functional.mse_loss(predicted_actions, actions))
            total_rmse += rmse.item()
            
            # 성공률 계산
            for threshold in success_thresholds:
                accuracy_key = f"accuracy_{int(threshold*100)}"
                # 모든 액션 차원에서 임계값 이내인지 확인
                within_threshold = torch.all(torch.abs(predicted_actions - actions) < threshold, dim=1)
                success_counts[accuracy_key] += within_threshold.sum().item()
            
            total_samples += images.shape[0]
            
            # 예측값과 타겟 저장
            predictions.extend(predicted_actions.cpu().numpy())
            targets.extend(actions.cpu().numpy())
    
    # 평균 계산
    avg_loss = total_loss / len(data_loader)
    avg_mae = total_mae / len(data_loader)
    avg_rmse = total_rmse / len(data_loader)
    
    # 성공률 계산
    success_rates = {}
    for key, count in success_counts.items():
        success_rates[key] = (count / total_samples) * 100
    
    # 예측값과 타겟을 numpy 배열로 변환
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # 각 액션 차원별 성능 분석
    action_dim_performance = {}
    for i, dim_name in enumerate(['linear_x', 'linear_y']):
        dim_predictions = predictions[:, i]
        dim_targets = targets[:, i]
        
        dim_mae = np.mean(np.abs(dim_predictions - dim_targets))
        dim_rmse = np.sqrt(np.mean((dim_predictions - dim_targets) ** 2))
        
        action_dim_performance[dim_name] = {
            'mae': float(dim_mae),
            'rmse': float(dim_rmse)
        }
    
    results = {
        'model_type': 'Optimized_2D_Action_Model',
        'total_samples': total_samples,
        'avg_loss': float(avg_loss),
        'avg_mae': float(avg_mae),
        'avg_rmse': float(avg_rmse),
        'success_rates': success_rates,
        'action_dim_performance': action_dim_performance,
        'predictions_shape': predictions.shape,
        'targets_shape': targets.shape
    }
    
    return results

def compare_with_existing_models():
    """기존 모델들과 성능 비교"""
    print("🔍 기존 모델들과 성능 비교 중...")
    
    # 기존 모델 체크포인트들 확인
    checkpoint_files = [
        'optimized_2d_action_model_best.pth',
        'fixed_robovlms_model_best.pth',
        'train_without_first_frame_model_best.pth',
        'train_fixed_robovlms_model_best.pth'
    ]
    
    existing_results = {}
    
    for checkpoint_file in checkpoint_files:
        if os.path.exists(checkpoint_file):
            try:
                checkpoint = torch.load(checkpoint_file, map_location='cpu')
                if 'val_loss' in checkpoint:
                    existing_results[checkpoint_file] = {
                        'val_loss': checkpoint['val_loss'],
                        'train_loss': checkpoint.get('train_loss', 'N/A'),
                        'epoch': checkpoint.get('epoch', 'N/A'),
                        'config': checkpoint.get('config', {})
                    }
                    print(f"📊 {checkpoint_file}: 검증 손실 = {checkpoint['val_loss']:.4f}")
            except Exception as e:
                print(f"❌ {checkpoint_file} 로드 실패: {e}")
    
    return existing_results

def main():
    """메인 평가 함수"""
    print("🎯 2D 액션 최적화 모델 평가 시작!")
    
    # 설정
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_path = '../../ROS_action/mobile_vla_dataset'
    batch_size = 8
    
    print(f"📊 설정: 디바이스={device}, 배치크기={batch_size}")
    
    # 프로세서 로드
    print("🔧 프로세서 로드 중...")
    processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
    
    # 평가 데이터셋 로드
    print("📊 평가 데이터셋 로드 중...")
    eval_dataset = Optimized2DEvaluationDataset(data_path, processor, 'eval')
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    
    # 모델 로드
    print("🤖 2D 액션 최적화 모델 로드 중...")
    model = Optimized2DActionModel(
        processor=processor,
        action_dim=2,
        dropout=0.2,
        use_claw_matrix=True,
        use_hierarchical=True,
        use_advanced_attention=True
    )
    
    # 체크포인트 로드
    checkpoint_path = 'optimized_2d_action_model_best.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 동적 어댑터 문제 해결을 위한 더미 포워드 패스
        dummy_input = torch.zeros(1, 3, 224, 224).to(device)
        with torch.no_grad():
            _ = model(dummy_input, "Navigate to target")
        
        # 호환되는 키만 로드
        model_state_dict = model.state_dict()
        checkpoint_state_dict = checkpoint['model_state_dict']
        
        # 호환되는 키만 필터링
        compatible_state_dict = {}
        for key in checkpoint_state_dict.keys():
            if key in model_state_dict and model_state_dict[key].shape == checkpoint_state_dict[key].shape:
                compatible_state_dict[key] = checkpoint_state_dict[key]
            else:
                print(f"⚠️ 호환되지 않는 키 건너뛰기: {key}")
        
        model.load_state_dict(compatible_state_dict, strict=False)
        print(f"✅ 체크포인트 로드 완료: {checkpoint_path}")
        print(f"   - 에포크: {checkpoint['epoch']}")
        print(f"   - 검증 손실: {checkpoint['val_loss']:.4f}")
        print(f"   - 로드된 파라미터: {len(compatible_state_dict)}/{len(checkpoint_state_dict)}")
    else:
        print(f"❌ 체크포인트를 찾을 수 없습니다: {checkpoint_path}")
        return
    
    model = model.to(device)
    
    # 모델 평가
    print("🎯 모델 평가 시작...")
    results = evaluate_2d_model(model, eval_loader, device)
    
    # 결과 출력
    print("\n" + "="*60)
    print("🎯 2D 액션 최적화 모델 평가 결과")
    print("="*60)
    print(f"📊 총 샘플 수: {results['total_samples']:,}")
    print(f"📊 평균 손실: {results['avg_loss']:.4f}")
    print(f"📊 평균 MAE: {results['avg_mae']:.4f}")
    print(f"📊 평균 RMSE: {results['avg_rmse']:.4f}")
    
    print("\n🎯 성공률:")
    for threshold, rate in results['success_rates'].items():
        print(f"   - {threshold}: {rate:.2f}%")
    
    print("\n📊 액션 차원별 성능:")
    for dim_name, performance in results['action_dim_performance'].items():
        print(f"   - {dim_name}:")
        print(f"     MAE: {performance['mae']:.4f}")
        print(f"     RMSE: {performance['rmse']:.4f}")
    
    # 기존 모델들과 비교
    print("\n" + "="*60)
    print("🔍 기존 모델들과 성능 비교")
    print("="*60)
    existing_results = compare_with_existing_models()
    
    if existing_results:
        print("\n📊 모델별 검증 손실 비교:")
        sorted_models = sorted(existing_results.items(), key=lambda x: x[1]['val_loss'])
        for i, (model_name, result) in enumerate(sorted_models):
            rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
            print(f"{rank} {model_name}: {result['val_loss']:.4f}")
    
    # 결과 저장
    results['existing_models_comparison'] = existing_results
    results['evaluation_config'] = {
        'device': device,
        'batch_size': batch_size,
        'data_path': data_path
    }
    
    with open('optimized_2d_action_evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 평가 결과 저장 완료: optimized_2d_action_evaluation_results.json")
    
    # 최종 요약
    print("\n" + "="*60)
    print("🎯 최종 요약")
    print("="*60)
    print(f"✅ 2D 액션 최적화 모델 평가 완료!")
    print(f"📊 주요 성능 지표:")
    print(f"   - 평균 MAE: {results['avg_mae']:.4f}")
    print(f"   - 평균 RMSE: {results['avg_rmse']:.4f}")
    print(f"   - 0.1 임계값 성공률: {results['success_rates']['accuracy_10']:.2f}%")
    print(f"   - 0.05 임계값 성공률: {results['success_rates']['accuracy_5']:.2f}%")
    print(f"   - 0.01 임계값 성공률: {results['success_rates']['accuracy_1']:.2f}%")

if __name__ == "__main__":
    main()
