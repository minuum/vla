import h5py
import cv2
import os
from pathlib import Path

def extract_images(h5_path, output_dir):
    """
    HDF5 파일에서 이미지를 추출하여 저장하는 스크립트
    """
    h5_path = Path(h5_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"H5 파일: {h5_path}")
    print(f"저장 경로: {output_dir}")
    
    try:
        with h5py.File(h5_path, 'r') as f:
            if 'images' not in f:
                print("오류: H5 파일에 'images' 데이터셋이 없습니다.")
                return
            
            # images 데이터셋 로드 (shape: [N, H, W, C])
            images = f['images'][:]
            num_frames = images.shape[0]
            print(f"총 {num_frames}개의 프레임을 추출합니다...")
            
            for i in range(num_frames):
                img = images[i]
                
                # H5 이미지는 일반적으로 RGB 형식이지만, OpenCV는 BGR을 사용하므로 변환이 필요할 수 있음
                # 만약 이미지가 이미 BGR이라면 변환을 건너뜀
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
                output_path = output_dir / f"frame_{i:03d}.png"
                cv2.imwrite(str(output_path), img_bgr)
                
            print(f"성공적으로 {num_frames}개의 이미지를 {output_dir}에 저장했습니다.")
    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    h5_file = "/home/billy/25-1kp/vla/ROS_action/basket_dataset/episode_20260129_010041_basket_1box_hori_left_core_medium.h5"
    # 결과물을 저장할 폴더 이름 (파일명 기반)
    folder_name = Path(h5_file).stem
    output_folder = Path("/home/billy/25-1kp/vla/ROS_action/extracted_images") / folder_name
    
    extract_images(h5_file, output_folder)
