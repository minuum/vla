#!/usr/bin/env python3
"""
Vision Resampler Enhanced 2D Action Model Evaluation Script
평가 메트릭: MAE, RMSE, 성공률 (개별 차원별, 전체)
"""

import torch
import torch.nn as nn
import numpy as np
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from transformers import AutoProcessor
from enhanced_2d_model_complete import Enhanced2DActionModel
from enhanced_dataset import create_enhanced_data_loaders

def evaluate_enhanced_model(model, test_loader, device, thresholds=[0.01, 0.05, 0.1, 0.2]):
    """
    향상된 모델 평가
    """
    model.eval()
    all_predictions = []
    all_targets = []
    total_loss = 0
    criterion = nn.MSELoss()
    
    print("🔍 모델 평가 중...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="평가 진행"):
            images = batch['image'].to(device)
            actions = batch['action'].to(device)
            texts = batch['text']
            
            predictions = model(images, texts)
            loss = criterion(predictions, actions)
            total_loss += loss.item()
            
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(actions.cpu().numpy())
    
    # 결과 통합
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # 기본 메트릭 계산
    mae = np.mean(np.abs(all_predictions - all_targets))
    rmse = np.sqrt(np.mean((all_predictions - all_targets) ** 2))
    
    # 차원별 성공률 계산
    success_rates = {}
    for threshold in thresholds:
        # 개별 차원별 성공률
        linear_x_success = np.mean(np.abs(all_predictions[:, 0] - all_targets[:, 0]) < threshold)
        linear_y_success = np.mean(np.abs(all_predictions[:, 1] - all_targets[:, 1]) < threshold)
        
        # 모든 차원이 동시에 성공하는 경우
        all_dims_success = np.mean(np.all(np.abs(all_predictions - all_targets) < threshold, axis=1))
        
        # 가중 평균 성공률 (linear_x에 더 높은 가중치)
        weighted_success = 0.7 * linear_x_success + 0.3 * linear_y_success
        
        success_rates[f'threshold_{threshold}'] = {
            'linear_x_success_rate': float(linear_x_success),
            'linear_y_success_rate': float(linear_y_success),
            'all_dims_success_rate': float(all_dims_success),
            'weighted_success_rate': float(weighted_success)
        }
    
    # 차원별 상세 분석
    dimension_analysis = {
        'linear_x': {
            'mae': float(np.mean(np.abs(all_predictions[:, 0] - all_targets[:, 0]))),
            'rmse': float(np.sqrt(np.mean((all_predictions[:, 0] - all_targets[:, 0]) ** 2))),
            'std': float(np.std(all_predictions[:, 0] - all_targets[:, 0])),
            'min_error': float(np.min(np.abs(all_predictions[:, 0] - all_targets[:, 0]))),
            'max_error': float(np.max(np.abs(all_predictions[:, 0] - all_targets[:, 0])))
        },
        'linear_y': {
            'mae': float(np.mean(np.abs(all_predictions[:, 1] - all_targets[:, 1]))),
            'rmse': float(np.sqrt(np.mean((all_predictions[:, 1] - all_targets[:, 1]) ** 2))),
            'std': float(np.std(all_predictions[:, 1] - all_targets[:, 1])),
            'min_error': float(np.min(np.abs(all_predictions[:, 1] - all_targets[:, 1]))),
            'max_error': float(np.max(np.abs(all_predictions[:, 1] - all_targets[:, 1])))
        }
    }
    
    return {
        'overall_metrics': {
            'mae': float(mae),
            'rmse': float(rmse),
            'avg_loss': float(total_loss / len(test_loader))
        },
        'success_rates': success_rates,
        'dimension_analysis': dimension_analysis,
        'sample_count': len(all_predictions)
    }

def main():
    parser = argparse.ArgumentParser(description='Vision Resampler Enhanced Model Evaluation')
    parser.add_argument('--model_path', type=str, required=True, help='모델 체크포인트 경로')
    parser.add_argument('--data_path', type=str, required=True, help='데이터셋 경로')
    parser.add_argument('--batch_size', type=int, default=8, help='배치 크기')
    parser.add_argument('--device', type=str, default='cuda', help='디바이스 (cuda/cpu)')
    parser.add_argument('--output_file', type=str, default='enhanced_model_evaluation_results.json', help='결과 저장 파일')
    
    args = parser.parse_args()
    
    # 디바이스 설정
    device = args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    print(f"🖥️  사용 디바이스: {device}")
    
    # 프로세서 로드
    processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
    
    # 모델 생성 및 로드
    print("📦 모델 로딩 중...")
    model = Enhanced2DActionModel(
        processor=processor,
        vision_dim=1024, language_dim=1024, action_dim=2, hidden_dim=512, dropout=0.2,
        use_vision_resampler=True
    )
    
    # 체크포인트 로드
    checkpoint = torch.load(args.model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    print(f"✅ 모델 로드 완료: {args.model_path}")
    
    # 데이터 로더 생성
    print("📊 데이터 로더 생성 중...")
    _, test_loader = create_enhanced_data_loaders(
        data_path=args.data_path, processor=processor, batch_size=args.batch_size,
        train_split=0.8, frame_selection='random', use_vision_resampler=True
    )
    
    print(f"📈 테스트 샘플 수: {len(test_loader.dataset)}")
    
    # 모델 평가
    results = evaluate_enhanced_model(model, test_loader, device)
    
    # 결과 출력
    print("\n" + "="*60)
    print("🎯 Vision Resampler Enhanced Model 평가 결과")
    print("="*60)
    
    print(f"\n📊 전체 성능:")
    print(f"   MAE: {results['overall_metrics']['mae']:.4f}")
    print(f"   RMSE: {results['overall_metrics']['rmse']:.4f}")
    print(f"   평균 손실: {results['overall_metrics']['avg_loss']:.4f}")
    print(f"   평가 샘플 수: {results['sample_count']}")
    
    print(f"\n🎯 성공률 (임계값별):")
    for threshold, rates in results['success_rates'].items():
        print(f"\n   임계값 {threshold.split('_')[1]}:")
        print(f"     Linear_X 성공률: {rates['linear_x_success_rate']:.1%}")
        print(f"     Linear_Y 성공률: {rates['linear_y_success_rate']:.1%}")
        print(f"     전체 차원 성공률: {rates['all_dims_success_rate']:.1%}")
        print(f"     가중 평균 성공률: {rates['weighted_success_rate']:.1%}")
    
    print(f"\n📈 차원별 상세 분석:")
    for dim, metrics in results['dimension_analysis'].items():
        print(f"\n   {dim.upper()}:")
        print(f"     MAE: {metrics['mae']:.4f}")
        print(f"     RMSE: {metrics['rmse']:.4f}")
        print(f"     표준편차: {metrics['std']:.4f}")
        print(f"     최소 오차: {metrics['min_error']:.4f}")
        print(f"     최대 오차: {metrics['max_error']:.4f}")
    
    # 결과 저장
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 결과가 {args.output_file}에 저장되었습니다.")
    
    # 성능 등급 평가
    mae = results['overall_metrics']['mae']
    if mae < 0.1:
        grade = "⭐⭐⭐⭐⭐ Excellent"
    elif mae < 0.2:
        grade = "⭐⭐⭐⭐ Good"
    elif mae < 0.3:
        grade = "⭐⭐⭐ Fair"
    elif mae < 0.5:
        grade = "⭐⭐ Poor"
    else:
        grade = "⭐ Very Poor"
    
    print(f"\n🏆 성능 등급: {grade} (MAE: {mae:.4f})")

if __name__ == "__main__":
    main()
