"""
Mobile-VLA H5 데이터셋에 Language Instruction 추가하는 스크립트

실제 태스크 명령어: "장애물을 피해 음료수 페트병 앞으로 도착해라"
영어 번역: "Navigate around obstacles and reach the front of the beverage bottle"
"""

import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm

# 태스크별 instruction 정의
TASK_INSTRUCTIONS = {
    # 기본 명령어 (실제 수집 태스크)
    'default': "Navigate around obstacles and reach the front of the beverage bottle",
    
    # 방향별 변형 (파일명 기반)
    'hori_left': "Navigate around obstacles and reach the front of the beverage bottle on the left",
    'hori_right': "Navigate around obstacles and reach the front of the beverage bottle on the right",
    
    # 시간대별 변형
    'evening': "Navigate around obstacles and reach the front of the beverage bottle in the evening",
    'morning': "Navigate around obstacles and reach the front of the beverage bottle in the morning",
}

def get_instruction_from_filename(filename: str) -> str:
    """파일명에서 적절한 instruction 추출"""
    filename_lower = filename.lower()
    
    # 우선순위: 방향 + 시간대
    if 'hori_left' in filename_lower and 'evening' in filename_lower:
        return "Navigate around obstacles and reach the front of the beverage bottle on the left in the evening"
    elif 'hori_right' in filename_lower and 'evening' in filename_lower:
        return "Navigate around obstacles and reach the front of the beverage bottle on the right in the evening"
    elif 'hori_left' in filename_lower:
        return TASK_INSTRUCTIONS['hori_left']
    elif 'hori_right' in filename_lower:
        return TASK_INSTRUCTIONS['hori_right']
    elif 'evening' in filename_lower:
        return TASK_INSTRUCTIONS['evening']
    else:
        return TASK_INSTRUCTIONS['default']

def add_language_instruction(h5_path: Path, instruction: str, dry_run: bool = False):
    """
    H5 파일에 language_instruction 추가
    
    Args:
        h5_path: H5 파일 경로
        instruction: 추가할 instruction 문자열
        dry_run: True면 실제로 추가하지 않고 출력만
    """
    if dry_run:
        print(f"[DRY RUN] {h5_path.name}")
        print(f"  Instruction: \"{instruction}\"")
        return
    
    try:
        with h5py.File(h5_path, 'a') as f:  # 'a' 모드로 열기 (추가/수정)
            # 이미 있으면 삭제 후 재생성
            if 'language_instruction' in f:
                del f['language_instruction']
            
            # UTF-8 인코딩하여 저장
            f.create_dataset(
                'language_instruction',
                data=np.array([instruction.encode('utf-8')], dtype='S256')
            )
            
        return True
    except Exception as e:
        print(f"❌ Error processing {h5_path.name}: {e}")
        return False

def process_all_files(data_dir: str, dry_run: bool = False):
    """모든 H5 파일에 language instruction 추가"""
    data_path = Path(data_dir)
    h5_files = sorted(list(data_path.glob('*.h5')))
    
    print(f"\n{'=' * 60}")
    print(f"📝 Adding Language Instructions to H5 files")
    print(f"{'=' * 60}")
    print(f"Directory: {data_dir}")
    print(f"Files found: {len(h5_files)}")
    print(f"Mode: {'DRY RUN (no changes)' if dry_run else 'WRITE MODE'}")
    print(f"{'=' * 60}\n")
    
    if dry_run:
        # Dry run: 처음 5개만 출력
        print("Preview (first 5 files):")
        for h5_file in h5_files[:5]:
            instruction = get_instruction_from_filename(h5_file.name)
            add_language_instruction(h5_file, instruction, dry_run=True)
        print(f"\n... and {len(h5_files) - 5} more files\n")
    else:
        # 실제 처리
        success_count = 0
        for h5_file in tqdm(h5_files, desc="Processing"):
            instruction = get_instruction_from_filename(h5_file.name)
            if add_language_instruction(h5_file, instruction, dry_run=False):
                success_count += 1
        
        print(f"\n✅ Successfully processed: {success_count}/{len(h5_files)} files")
    
    # 검증: 첫 번째 파일 확인
    if not dry_run and len(h5_files) > 0:
        print(f"\n{'=' * 60}")
        print("🔍 Verification (first file)")
        print(f"{'=' * 60}")
        verify_file(h5_files[0])

def verify_file(h5_path: Path):
    """파일 검증"""
    with h5py.File(h5_path, 'r') as f:
        print(f"\nFile: {h5_path.name}")
        print(f"Keys: {list(f.keys())}")
        
        if 'language_instruction' in f:
            instruction_data = f['language_instruction'][0]
            instruction = instruction_data.decode('utf-8')
            print(f"\n✅ Language Instruction:")
            print(f"   \"{instruction}\"")
        else:
            print("\n❌ No language_instruction found!")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Add language instructions to Mobile-VLA H5 files')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/Users/minu/dev/vla/ROS_action/mobile_vla_dataset',
        help='Directory containing H5 files'
    )
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Preview changes without modifying files'
    )
    
    args = parser.parse_args()
    
    # 실행
    process_all_files(args.data_dir, dry_run=args.dry_run)
    
    print(f"\n{'=' * 60}")
    print("📋 Next Steps:")
    print(f"{'=' * 60}")
    print("1. Run with --dry_run to preview")
    print("2. Run without --dry_run to actually add instructions")
    print("3. Verify in your training script:")
    print("   with h5py.File(path, 'r') as f:")
    print("       instruction = f['language_instruction'][0].decode('utf-8')")
    print(f"{'=' * 60}\n")

if __name__ == "__main__":
    main()
