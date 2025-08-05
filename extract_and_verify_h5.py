#!/usr/bin/env python3
import h5py
import numpy as np
import cv2
import argparse
from pathlib import Path

def check_h5_file(file_path: Path):
    """HDF5 파일의 메타데이터와 데이터 구조를 출력합니다."""
    if not file_path.is_file():
        print(f"❌ 파일이 존재하지 않습니다: {file_path}")
        return

    try:
        with h5py.File(file_path, 'r') as f:
            print(f"📁 파일: {file_path.name}")
            print(f"💾 크기: {file_path.stat().st_size / (1024*1024):.2f} MB")
            print("="*50)
            
            print("📋 메타데이터:")
            for key, value in f.attrs.items():
                print(f"   {key}: {value}")
            
            print("\n📦 데이터 구조:")
            for name, dset in f.items():
                print(f"   📄 {name}: {dset.shape} {dset.dtype}")

            if 'action_chunks' not in f:
                print("\n❌ Action Chunks 없음")

    except Exception as e:
        print(f"HDF5 파일을 읽는 중 오류 발생: {e}")

def extract_images(file_path: Path, output_dir: Path):
    """HDF5 파일에서 이미지를 추출하여 PNG 파일로 저장합니다."""
    if not file_path.is_file():
        print(f"❌ 파일이 존재하지 않습니다: {file_path}")
        return

    output_dir.mkdir(exist_ok=True)
    print(f"\n🖼️  'images' 데이터셋을 '{output_dir}' 폴더에 추출합니다...")

    try:
        with h5py.File(file_path, 'r') as f:
            if 'images' not in f:
                print("'images' 데이터셋을 찾을 수 없습니다.")
                return

            images = f['images']
            num_images = images.shape[0]

            for i in range(num_images):
                # 데이터셋의 이미지는 BGR 형식이므로 별도 변환 없이 저장합니다.
                img_bgr = images[i]
                save_path = output_dir / f"frame_{i:04d}.png"
                cv2.imwrite(str(save_path), img_bgr)
                print(f"\r   -> 저장 중... {i+1}/{num_images}", end="")
            print("\n✅ 추출 완료!")

    except Exception as e:
        print(f"이미지 추출 중 오류 발생: {e}")

def save_single_image(file_path: Path, index: int):
    """HDF5 파일에서 특정 인덱스의 이미지를 파일로 저장합니다."""
    if not file_path.is_file():
        print(f"❌ 파일이 존재하지 않습니다: {file_path}")
        return

    try:
        with h5py.File(file_path, 'r') as f:
            if 'images' not in f:
                print("'images' 데이터셋을 찾을 수 없습니다.")
                return

            images = f['images']
            if not (0 <= index < images.shape[0]):
                print(f"❌ 인덱스 오류: 0에서 {images.shape[0]-1} 사이의 값을 입력하세요.")
                return

            img_bgr = images[index]
            save_path = file_path.parent / f"viewed_{file_path.stem}_frame_{index}.png"
            cv2.imwrite(str(save_path), img_bgr)
            print(f"\n🖼️  프레임 {index}번 이미지를 '{save_path}' 파일로 저장했습니다.")

    except Exception as e:
        print(f"이미지를 저장하는 중 오류 발생: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HDF5 데이터셋 파일을 확인하고 이미지를 추출/저장합니다.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "file_path", 
        type=Path, 
        help="분석할 .h5 파일 경로"
    )
    parser.add_argument(
        "--extract",
        nargs='?',
        const=True,
        default=False,
        help="'images' 데이터셋의 모든 이미지를 추출합니다.\n(기본값: 파일명과 동일한 이름의 폴더에 저장)\n(사용법: --extract [저장할_폴더명])"
    )
    parser.add_argument(
        "--view", 
        type=int, 
        metavar="FRAME_INDEX",
        help="지정된 인덱스의 프레임을 png 파일로 저장합니다."
    )

    args = parser.parse_args()

    check_h5_file(args.file_path)

    if args.extract:
        output_dir = Path(args.extract) if isinstance(args.extract, str) else args.file_path.parent / args.file_path.stem
        extract_images(args.file_path, output_dir)

    if args.view is not None:
        save_single_image(args.file_path, args.view)