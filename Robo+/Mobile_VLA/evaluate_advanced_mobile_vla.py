#!/usr/bin/env python3
"""
📊 Advanced Mobile VLA Model 성능 평가
MAE, RMSE, 예측 정확도 계산 및 이전 모델과 비교
"""
import torch
import torch.nn as nn
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd

# RoboVLMs 모듈 추가
sys.path.append(str(Path(__file__).parent / "robovlms" / "models"))
from advanced_mobile_vla_model import AdvancedMobileVLAModel
from train_advanced_mobile_vla import MobileVLADataset, collate_fn
from torch.utils.data import DataLoader

def calculate_mae_rmse(predictions, targets):
    """MAE와 RMSE 계산"""
    mae = mean_absolute_error(targets.flatten(), predictions.flatten())
    rmse = np.sqrt(mean_squared_error(targets.flatten(), predictions.flatten()))
    return mae, rmse

def calculate_prediction_accuracy(predictions, targets, threshold=0.1):
    """예측 정확도 계산 (임계값 내 오차 비율)"""
    abs_error = np.abs(predictions - targets)
    within_threshold = (abs_error <= threshold).sum()
    total_predictions = abs_error.size
    accuracy = within_threshold / total_predictions * 100
    return accuracy

def evaluate_advanced_mobile_vla():
    """Advanced Mobile VLA 모델 성능 평가"""
    print("🚀 Advanced Mobile VLA 모델 성능 평가 시작")
    
    # 설정
    config = {
        'data_paths': [
            '../../ROS_action/mobile_vla_dataset',
            'augmented_dataset',
            'distance_aware_augmented_dataset'
        ],
        'batch_size': 1,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'model_path': 'advanced_mobile_vla_best.pth'
    }
    
    print(f"💻 디바이스: {config['device']}")
    
    # 데이터셋 로드 (검증용으로만 사용)
    print("📊 데이터셋 로딩 중...")
    dataset = MobileVLADataset(config['data_paths'])
    
    # 검증 데이터셋 (전체의 20%)
    val_size = max(1, int(len(dataset) * 0.2))
    train_size = len(dataset) - val_size
    _, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)
    
    print(f"📈 검증 데이터: {len(val_dataset)}개 에피소드")
    
    # 모델 로드
    print("🤖 모델 로딩 중...")
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
    
    model = AdvancedMobileVLAModel(
        processor=processor,
        vision_dim=768,
        language_dim=768,
        action_dim=3,
        fusion_dim=512,
        plan_dim=256,
        num_claw_layers=3,
        num_subgoals=6,
        frames_per_subgoal=3,
        use_claw_matrix=True,
        use_hierarchical=True,
        use_advanced_attention=True
    ).to(config['device'])
    
    # 모델 가중치 로드
    if Path(config['model_path']).exists():
        checkpoint = torch.load(config['model_path'], map_location=config['device'])
        
        # 체크포인트 형식 확인
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ 체크포인트 모델 로드 완료: {config['model_path']}")
            print(f"   에포크: {checkpoint.get('epoch', 'N/A')}")
            print(f"   훈련 손실: {checkpoint.get('train_loss', 'N/A'):.4f}")
            print(f"   검증 손실: {checkpoint.get('val_loss', 'N/A'):.4f}")
        else:
            model.load_state_dict(checkpoint)
            print(f"✅ 모델 로드 완료: {config['model_path']}")
    else:
        print(f"❌ 모델 파일 없음: {config['model_path']}")
        return
    
    model.eval()
    
    # 성능 평가
    print("📊 성능 평가 중...")
    all_predictions = []
    all_targets = []
    all_mae_per_episode = []
    all_rmse_per_episode = []
    all_accuracy_per_episode = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="평가 진행")):
            try:
                images = batch['images'].to(config['device']).float()
                actions = batch['actions'].to(config['device']).float()
                
                # 거리 라벨 생성
                batch_size = images.shape[0]
                distance_labels = torch.randint(0, 3, (batch_size,), device=config['device']).long()
                
                # 예측
                predicted_actions = model(
                    images=images,
                    distance_labels=distance_labels
                )
                
                # 타겟 액션 맞추기
                target_actions = actions[:, :predicted_actions.shape[1], :predicted_actions.shape[2]]
                
                # numpy로 변환
                pred_np = predicted_actions.cpu().numpy()
                target_np = target_actions.cpu().numpy()
                
                # 에피소드별 성능 계산
                for i in range(pred_np.shape[0]):
                    pred_episode = pred_np[i]
                    target_episode = target_np[i]
                    
                    mae, rmse = calculate_mae_rmse(pred_episode, target_episode)
                    accuracy = calculate_prediction_accuracy(pred_episode, target_episode)
                    
                    all_mae_per_episode.append(mae)
                    all_rmse_per_episode.append(rmse)
                    all_accuracy_per_episode.append(accuracy)
                    
                    all_predictions.append(pred_episode.flatten())
                    all_targets.append(target_episode.flatten())
                
            except Exception as e:
                print(f"⚠️ 배치 {batch_idx} 처리 중 오류: {str(e)}")
                continue
    
    # 전체 성능 계산
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)
    
    overall_mae = mean_absolute_error(all_targets, all_predictions)
    overall_rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
    overall_accuracy = calculate_prediction_accuracy(all_predictions, all_targets)
    
    # 에피소드별 평균
    avg_mae = np.mean(all_mae_per_episode)
    avg_rmse = np.mean(all_rmse_per_episode)
    avg_accuracy = np.mean(all_accuracy_per_episode)
    
    # 결과 출력
    print("\n" + "="*60)
    print("📊 Advanced Mobile VLA 모델 성능 평가 결과")
    print("="*60)
    print(f"🎯 전체 MAE: {overall_mae:.6f}")
    print(f"🎯 전체 RMSE: {overall_rmse:.6f}")
    print(f"🎯 전체 예측 정확도 (0.1 임계값): {overall_accuracy:.2f}%")
    print(f"📈 에피소드별 평균 MAE: {avg_mae:.6f}")
    print(f"📈 에피소드별 평균 RMSE: {avg_rmse:.6f}")
    print(f"📈 에피소드별 평균 정확도: {avg_accuracy:.2f}%")
    print(f"📊 평가 데이터 수: {len(val_dataset)}개 에피소드")
    print(f"📊 총 예측 수: {len(all_predictions)}개")
    
    # 결과 저장
    results = {
        'model_name': 'Advanced Mobile VLA (Claw Matrix + Hierarchical Planning + Advanced Attention)',
        'overall_mae': float(overall_mae),
        'overall_rmse': float(overall_rmse),
        'overall_accuracy': float(overall_accuracy),
        'avg_episode_mae': float(avg_mae),
        'avg_episode_rmse': float(avg_rmse),
        'avg_episode_accuracy': float(avg_accuracy),
        'num_episodes': len(val_dataset),
        'total_predictions': len(all_predictions),
        'evaluation_date': pd.Timestamp.now().isoformat()
    }
    
    with open('advanced_mobile_vla_evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 결과 저장: advanced_mobile_vla_evaluation_results.json")
    
    return results

if __name__ == "__main__":
    evaluate_advanced_mobile_vla()
