#!/usr/bin/env python3
import h5py
import numpy as np
import cv2
import argparse
import json
import pandas as pd
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
                print("\n💡 정보: Action Chunks 데이터가 없습니다 (이미지 추출에는 영향 없음)")

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

def export_to_csv(file_path: Path, output_path: Path = None):
    """HDF5 데이터를 CSV 파일로 추출합니다."""
    if not file_path.is_file():
        print(f"❌ 파일이 존재하지 않습니다: {file_path}")
        return

    try:
        with h5py.File(file_path, 'r') as f:
            # 메타데이터와 액션 데이터 추출
            metadata = dict(f.attrs)
            actions = f['actions'][:]
            action_event_types = f['action_event_types'][:]
            
            # DataFrame 생성
            data = []
            for i in range(len(actions)):
                row = {
                    'frame_index': i,
                    'action_x': actions[i][0],
                    'action_y': actions[i][1], 
                    'action_z': actions[i][2],
                    'event_type': action_event_types[i].decode('utf-8') if isinstance(action_event_types[i], bytes) else str(action_event_types[i]),
                    'episode_name': metadata.get('episode_name', ''),
                    'total_duration': metadata.get('total_duration', 0),
                    'action_chunk_size': metadata.get('action_chunk_size', 0)
                }
                data.append(row)
            
            df = pd.DataFrame(data)
            
            # 출력 파일 경로 설정
            if output_path is None:
                # H5 파일의 time_period 메타데이터를 읽어서 파일명에 추가
                time_period = metadata.get('time_period', None)
                stem = file_path.stem
                
                # stem에서 "medium" 뒤에 시간대 정보 추가
                if time_period and 'medium' in stem:
                    # medium 뒤에 시간대 추가
                    stem = stem.replace('medium', f'medium_{time_period}')
                elif time_period:
                    # medium이 없으면 그냥 끝에 추가
                    stem = f"{stem}_{time_period}"
                
                output_path = file_path.parent / f"{stem}_data.csv"
            
            df.to_csv(output_path, index=False)
            print(f"📊 CSV 파일 저장 완료: {output_path}")
            print(f"   총 {len(data)}개 프레임 데이터 추출")
            
    except Exception as e:
        print(f"CSV 추출 중 오류 발생: {e}")

def export_to_json(file_path: Path, output_path: Path = None):
    """HDF5 데이터를 JSON 파일로 추출합니다."""
    if not file_path.is_file():
        print(f"❌ 파일이 존재하지 않습니다: {file_path}")
        return

    try:
        with h5py.File(file_path, 'r') as f:
            # 전체 데이터 구조 생성
            metadata = {}
            for key, value in f.attrs.items():
                if isinstance(value, (np.integer, np.floating)):
                    metadata[key] = value.item()
                else:
                    metadata[key] = value
            
            data = {
                "file_name": file_path.name,
                "file_size_mb": float(file_path.stat().st_size / (1024*1024)),
                "metadata": metadata,
                "frames": []
            }
            
            # 각 프레임별 데이터
            actions = f['actions'][:]
            action_event_types = f['action_event_types'][:]
            
            for i in range(len(actions)):
                frame_data = {
                    "frame_index": i,
                    "action": {
                        "x": float(actions[i][0]),
                        "y": float(actions[i][1]), 
                        "z": float(actions[i][2])
                    },
                    "event_type": action_event_types[i].decode('utf-8') if isinstance(action_event_types[i], bytes) else str(action_event_types[i]),
                    "image_file": f"frame_{i:04d}.png"
                }
                data["frames"].append(frame_data)
            
            # 출력 파일 경로 설정
            if output_path is None:
                output_path = file_path.parent / f"{file_path.stem}_data.json"
            
            with open(output_path, 'w', encoding='utf-8') as json_file:
                json.dump(data, json_file, indent=2, ensure_ascii=False)
            
            print(f"📄 JSON 파일 저장 완료: {output_path}")
            print(f"   총 {len(data['frames'])}개 프레임 데이터 추출")
            
    except Exception as e:
        print(f"JSON 추출 중 오류 발생: {e}")


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
    parser.add_argument(
        "--csv",
        action="store_true",
        help="액션 데이터를 CSV 파일로 추출합니다."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="전체 데이터를 JSON 파일로 추출합니다."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="이미지, CSV, JSON을 모두 추출합니다."
    )

    args = parser.parse_args()

    check_h5_file(args.file_path)

    if args.extract or args.all:
        output_dir = Path(args.extract) if isinstance(args.extract, str) else args.file_path.parent / args.file_path.stem
        extract_images(args.file_path, output_dir)

    if args.csv or args.all:
        export_to_csv(args.file_path)

    if args.json or args.all:
        export_to_json(args.file_path)

    if args.view is not None:
        save_single_image(args.file_path, args.view)