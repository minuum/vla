import h5py
import cv2
import os
import numpy as np

# 경로 설정
DATASET_DIR = "/home/billy/25-1kp/vla/ROS_action/basket_dataset_v2"
OUTPUT_DIR = "/home/billy/25-1kp/vla/debug_extracted_images"

# 출력 디렉토리 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 대상 파일 리스트 (처음 3개)
h5_files = [f for f in os.listdir(DATASET_DIR) if f.endswith('.h5')][:3]

print(f"대상 파일: {h5_files}")

for h5_file in h5_files:
    file_path = os.path.join(DATASET_DIR, h5_file)
    episode_name = h5_file.replace('.h5', '')
    episode_dir = os.path.join(OUTPUT_DIR, episode_name)
    os.makedirs(episode_dir, exist_ok=True)
    
    try:
        with h5py.File(file_path, 'r') as f:
            if 'images' in f:
                num_frames = f['images'].shape[0]
                print(f"[{episode_name}] 총 {num_frames} 프레임 추출 시작...")
                
                for idx in range(num_frames):
                    img = f['images'][idx]
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    output_name = f"frame_{idx:03d}.jpg"
                    output_path = os.path.join(episode_dir, output_name)
                    cv2.imwrite(output_path, img_bgr)
                
                print(f"[{episode_name}] 추출 완료.")
            else:
                print(f"Error: 'images' 키를 찾을 수 없음 - {h5_file}")
    except Exception as e:
        print(f"Error 처리 중 {h5_file}: {e}")

print(f"\n모든 작업 완료. 결과는 {OUTPUT_DIR} 에 저장되었습니다.")
