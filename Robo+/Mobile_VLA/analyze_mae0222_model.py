#!/usr/bin/env python3
"""
실제 MAE 0.222 모델 구조 분석
체크포인트에서 정확한 모델 구조 파악
"""

import torch
import json
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_mae0222_model():
    """MAE 0.222 모델 구조 분석"""
    logger.info("🔍 MAE 0.222 모델 구조 분석 시작")
    
    # 체크포인트 로드
    checkpoint_path = "results/simple_lstm_results_extended/best_simple_lstm_model.pth"
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        logger.info(f"✅ 체크포인트 로드 성공: {checkpoint_path}")
        
        # 기본 정보 출력
        logger.info(f"   - 에포크: {checkpoint.get('epoch', 'N/A')}")
        logger.info(f"   - 검증 MAE: {checkpoint.get('val_mae', 'N/A')}")
        
        # 모델 상태 딕셔너리 분석
        state_dict = checkpoint.get('model_state_dict', {})
        logger.info(f"   - 모델 키 수: {len(state_dict)}")
        
        # 키들을 카테고리별로 분류
        kosmos_keys = []
        rnn_keys = []
        action_keys = []
        other_keys = []
        
        for key in state_dict.keys():
            if 'kosmos' in key.lower():
                kosmos_keys.append(key)
            elif 'rnn' in key.lower():
                rnn_keys.append(key)
            elif 'action' in key.lower():
                action_keys.append(key)
            else:
                other_keys.append(key)
        
        logger.info(f"\n📊 모델 구조 분석:")
        logger.info(f"   - Kosmos2 관련 키: {len(kosmos_keys)}개")
        logger.info(f"   - RNN 관련 키: {len(rnn_keys)}개")
        logger.info(f"   - Action 관련 키: {len(action_keys)}개")
        logger.info(f"   - 기타 키: {len(other_keys)}개")
        
        # RNN 구조 분석
        logger.info(f"\n🧠 RNN 구조 분석:")
        for key in rnn_keys:
            shape = state_dict[key].shape
            logger.info(f"   - {key}: {shape}")
        
        # Action Head 구조 분석
        logger.info(f"\n🎯 Action Head 구조 분석:")
        for key in action_keys:
            shape = state_dict[key].shape
            logger.info(f"   - {key}: {shape}")
        
        # Kosmos2 구조 분석 (처음 10개만)
        logger.info(f"\n🖼️ Kosmos2 구조 분석 (처음 10개):")
        for i, key in enumerate(kosmos_keys[:10]):
            shape = state_dict[key].shape
            logger.info(f"   - {key}: {shape}")
        
        if len(kosmos_keys) > 10:
            logger.info(f"   ... (총 {len(kosmos_keys)}개 키)")
        
        # 입력 크기 추정
        rnn_input_size = None
        for key in rnn_keys:
            if 'weight_ih_l0' in key:
                rnn_input_size = state_dict[key].shape[1]
                break
        
        logger.info(f"\n🔍 구조 추정:")
        logger.info(f"   - RNN 입력 크기: {rnn_input_size}")
        logger.info(f"   - RNN 히든 크기: 4096 (추정)")
        logger.info(f"   - RNN 레이어 수: 4 (추정)")
        logger.info(f"   - 출력 크기: 2 (linear_x, linear_y)")
        
        # 결과 저장
        analysis_result = {
            'checkpoint_info': {
                'epoch': checkpoint.get('epoch'),
                'val_mae': checkpoint.get('val_mae'),
                'total_keys': len(state_dict)
            },
            'model_structure': {
                'kosmos_keys_count': len(kosmos_keys),
                'rnn_keys_count': len(rnn_keys),
                'action_keys_count': len(action_keys),
                'other_keys_count': len(other_keys),
                'rnn_input_size': rnn_input_size,
                'rnn_hidden_size': 4096,
                'rnn_layers': 4,
                'output_size': 2
            },
            'key_categories': {
                'kosmos_keys': kosmos_keys[:20],  # 처음 20개만
                'rnn_keys': rnn_keys,
                'action_keys': action_keys,
                'other_keys': other_keys[:20]  # 처음 20개만
            }
        }
        
        with open('mae0222_model_analysis.json', 'w') as f:
            json.dump(analysis_result, f, indent=2)
        
        logger.info(f"\n✅ 분석 결과가 mae0222_model_analysis.json에 저장되었습니다")
        
        return analysis_result
        
    except Exception as e:
        logger.error(f"❌ 체크포인트 분석 실패: {e}")
        return None

def main():
    result = analyze_mae0222_model()
    return result

if __name__ == "__main__":
    main()
