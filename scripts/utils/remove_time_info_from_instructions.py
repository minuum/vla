"""
기존 H5 파일의 language_instruction에서 시간 정보 제거 스크립트
"""

import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm

def remove_time_info_from_instruction(instruction: str) -> str:
    """instruction에서 시간 정보 제거"""
    # 제거할 시간 관련 문구들
    time_phrases = [
        " in the evening",
        " in the morning",
        " at night",
        " at dawn"
    ]
    
    cleaned = instruction
    for phrase in time_phrases:
        cleaned = cleaned.replace(phrase, "")
    
    return cleaned

def update_h5_instructions(data_dir: str, dry_run: bool = False):
    """모든 H5 파일의 language_instruction에서 시간 정보 제거"""
    data_path = Path(data_dir)
    h5_files = sorted(list(data_path.glob('*.h5')))
    
    print(f"\n{'=' * 60}")
    print(f"🔄 Removing time information from language instructions")
    print(f"{'=' * 60}")
    print(f"Directory: {data_dir}")
    print(f"Files found: {len(h5_files)}")
    print(f"Mode: {'DRY RUN (no changes)' if dry_run else 'WRITE MODE'}")
    print(f"{'=' * 60}\n")
    
    if dry_run:
        # Preview
        print("Preview (first 5 files):")
        for h5_file in h5_files[:5]:
            with h5py.File(h5_file, 'r') as f:
                if 'language_instruction' in f:
                    old_inst = f['language_instruction'][0].decode('utf-8')
                    new_inst = remove_time_info_from_instruction(old_inst)
                    if old_inst != new_inst:
                        print(f"\n{h5_file.name}")
                        print(f"  Before: \"{old_inst}\"")
                        print(f"  After:  \"{new_inst}\"")
                    else:
                        print(f"\n{h5_file.name} - No change needed")
        print(f"\n... and {len(h5_files) - 5} more files\n")
    else:
        # Actual update
        updated_count = 0
        unchanged_count = 0
        
        for h5_file in tqdm(h5_files, desc="Processing"):
            try:
                with h5py.File(h5_file, 'a') as f:
                    if 'language_instruction' in f:
                        old_inst = f['language_instruction'][0].decode('utf-8')
                        new_inst = remove_time_info_from_instruction(old_inst)
                        
                        if old_inst != new_inst:
                            # Update instruction
                            del f['language_instruction']
                            f.create_dataset(
                                'language_instruction',
                                data=np.array([new_inst.encode('utf-8')], dtype='S256')
                            )
                            updated_count += 1
                        else:
                            unchanged_count += 1
            except Exception as e:
                print(f"\n❌ Error processing {h5_file.name}: {e}")
        
        print(f"\n✅ Updated: {updated_count} files")
        print(f"📊 Unchanged: {unchanged_count} files")
        print(f"📝 Total: {len(h5_files)} files")
        
        # Verification
        if updated_count > 0:
            print(f"\n{'=' * 60}")
            print("🔍 Verification (first updated file)")
            print(f"{'=' * 60}")
            for h5_file in h5_files:
                with h5py.File(h5_file, 'r') as f:
                    if 'language_instruction' in f:
                        inst = f['language_instruction'][0].decode('utf-8')
                        print(f"\nFile: {h5_file.name}")
                        print(f"Language Instruction: \"{inst}\"")
                        break

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Remove time info from language instructions in H5 files')
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
    update_h5_instructions(args.data_dir, dry_run=args.dry_run)
    
    print(f"\n{'=' * 60}")
    print("✅ Complete!")
    print(f"{'=' * 60}\n")

if __name__ == "__main__":
    main()
