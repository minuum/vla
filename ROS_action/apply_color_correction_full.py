#!/usr/bin/env python3
"""
전체 데이터셋 색상 보정 적용
Method 3 (R/G=1.05) 사용하여 모든 H5 파일 보정
"""

import cv2
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
import json
import shutil


def correct_color_method3(img, target_rg=1.05):
    """Method 3: R/G 비율을 1.05로 조정"""
    b, g, r = cv2.split(img)
    
    current_rg = np.mean(r) / (np.mean(g) + 1e-6)
    scale_r = target_rg / (current_rg + 1e-6)
    
    r_corrected = np.clip(r * scale_r, 0, 255).astype(np.uint8)
    
    return cv2.merge([b, g, r_corrected])


def process_h5_file(input_h5_path: Path, output_h5_path: Path):
    """H5 파일 읽어서 색상 보정 후 저장"""
    
    try:
        with h5py.File(input_h5_path, 'r') as f_in:
            # H5 구조 확인
            if 'images' not in f_in:
                return {'error': 'No images dataset found'}
            
            # 모든 이미지 읽기
            images = f_in['images'][:]
            actions = f_in['actions'][:] if 'actions' in f_in else None
            
            # 이미지 보정
            corrected_images = []
            for img_rgb in images:
                # RGB to BGR for processing
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                
                # 보정 적용
                img_corrected_bgr = correct_color_method3(img_bgr, target_rg=1.05)
                
                # BGR to RGB for storage
                img_corrected_rgb = cv2.cvtColor(img_corrected_bgr, cv2.COLOR_BGR2RGB)
                
                corrected_images.append(img_corrected_rgb)
            
            corrected_images = np.array(corrected_images)
        
        # 새 H5 파일 생성
        output_h5_path.parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(output_h5_path, 'w') as f_out:
            f_out.create_dataset('images', data=corrected_images, compression='gzip')
            if actions is not None:
                f_out.create_dataset('actions', data=actions, compression='gzip')
        
        return {
            'success': True,
            'frames': len(corrected_images)
        }
        
    except Exception as e:
        return {'error': str(e)}


def apply_correction_to_dataset():
    """전체 데이터셋에 색상 보정 적용"""
    
    source_dir = Path(__file__).parent / 'mobile_vla_dataset'
    output_dir = Path(__file__).parent / 'mobile_vla_dataset_corrected'
    
    print("🎨 전체 데이터셋 색상 보정 시작")
    print(f"   방법: Method 3 (R/G=1.05)")
    print(f"   원본: {source_dir}")
    print(f"   출력: {output_dir}")
    print("=" * 80)
    
    # 모든 H5 파일 찾기
    h5_files = sorted(source_dir.glob("**/*.h5"))
    
    print(f"총 {len(h5_files)}개 H5 파일 발견")
    
    if not h5_files:
        print("❌ H5 파일을 찾을 수 없습니다!")
        return
    
    # 진행 상황
    results = []
    success_count = 0
    error_count = 0
    
    for h5_path in tqdm(h5_files, desc="색상 보정 적용"):
        # 출력 경로
        relative_path = h5_path.relative_to(source_dir)
        output_path = output_dir / relative_path
        
        # 보정 적용
        result = process_h5_file(h5_path, output_path)
        
        if 'success' in result:
            success_count += 1
        else:
            error_count += 1
            print(f"\n⚠️  오류: {h5_path.name} - {result.get('error', 'Unknown')}")
        
        results.append({
            'input': str(h5_path),
            'output': str(output_path),
            'result': result
        })
    
    # JSON 파일도 복사
    print("\n📋 JSON 파일 복사 중...")
    json_files = list(source_dir.glob("**/*.json"))
    for json_path in json_files:
        relative_path = json_path.relative_to(source_dir)
        output_path = output_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(json_path, output_path)
    
    # 요약
    print("\n" + "=" * 80)
    print("📊 보정 완료")
    print("=" * 80)
    print(f"성공: {success_count}")
    print(f"실패: {error_count}")
    print(f"JSON 파일: {len(json_files)}개 복사")
    
    # 결과 저장
    summary_file = output_dir / 'correction_summary.json'
    with open(summary_file, 'w') as f:
        json.dump({
            'method': 'Method 3 (R/G=1.05)',
            'total_files': len(h5_files),
            'success': success_count,
            'errors': error_count,
            'results': results
        }, f, indent=2)
    
    print(f"\n💾 상세 결과: {summary_file}")
    print(f"\n✅ 색상 보정 완료!")
    print(f"   새 데이터셋: {output_dir}")
    
    return output_dir


if __name__ == '__main__':
    output_dir = apply_correction_to_dataset()
    
    if output_dir:
        print(f"\n🎉 {output_dir.name} 디렉토리가 생성되었습니다!")
        print(f"   - 모든 이미지가 Method 3으로 보정됨")
        print(f"   - R/G 비율: 1.05 (따뜻한 톤)")
        print(f"   - 초록 뚜껑이 더 잘 보임")
