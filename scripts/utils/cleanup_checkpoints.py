#!/usr/bin/env python3
"""
체크포인트 자동 정리 스크립트
- 최근 N개의 체크포인트만 유지
- 오래된 체크포인트 자동 삭제
- 디스크 공간 확보
"""

import os
import glob
import argparse
from pathlib import Path
from datetime import datetime


def get_checkpoint_info(ckpt_path):
    """체크포인트 파일 정보 가져오기"""
    stat = os.stat(ckpt_path)
    return {
        'path': ckpt_path,
        'mtime': stat.st_mtime,
        'size': stat.st_size,
        'size_mb': stat.st_size / (1024 * 1024)
    }


def cleanup_checkpoints(runs_dir, keep_last=3, dry_run=False):
    """
    체크포인트 정리
    
    Args:
        runs_dir: runs 디렉토리 경로
        keep_last: 유지할 최근 체크포인트 개수
        dry_run: True면 삭제하지 않고 출력만
    """
    # 모든 체크포인트 찾기
    ckpt_pattern = os.path.join(runs_dir, "**", "*.ckpt")
    ckpt_files = glob.glob(ckpt_pattern, recursive=True)
    
    if not ckpt_files:
        print("❌ 체크포인트 파일을 찾을 수 없습니다.")
        return
    
    print(f"📁 총 체크포인트 파일: {len(ckpt_files)}개")
    
    # 체크포인트 정보 수집
    ckpt_infos = [get_checkpoint_info(f) for f in ckpt_files]
    
    # 수정 시간으로 정렬 (최신순)
    ckpt_infos.sort(key=lambda x: x['mtime'], reverse=True)
    
    # 총 크기 계산
    total_size_gb = sum(info['size'] for info in ckpt_infos) / (1024**3)
    print(f"💾 총 크기: {total_size_gb:.2f} GB")
    print()
    
    # 유지할 체크포인트와 삭제할 체크포인트 분리
    to_keep = ckpt_infos[:keep_last]
    to_delete = ckpt_infos[keep_last:]
    
    print(f"✅ 유지할 체크포인트 ({len(to_keep)}개):")
    for info in to_keep:
        mtime = datetime.fromtimestamp(info['mtime']).strftime('%Y-%m-%d %H:%M:%S')
        print(f"  - {Path(info['path']).name}: {info['size_mb']:.1f} MB (수정: {mtime})")
    print()
    
    if not to_delete:
        print("✅ 삭제할 체크포인트가 없습니다.")
        return
    
    print(f"🗑️  삭제할 체크포인트 ({len(to_delete)}개):")
    delete_size_gb = sum(info['size'] for info in to_delete) / (1024**3)
    for info in to_delete:
        mtime = datetime.fromtimestamp(info['mtime']).strftime('%Y-%m-%d %H:%M:%S')
        print(f"  - {Path(info['path']).name}: {info['size_mb']:.1f} MB (수정: {mtime})")
    print()
    print(f"💾 확보될 공간: {delete_size_gb:.2f} GB")
    print()
    
    if dry_run:
        print("🔍 DRY RUN 모드: 실제로 삭제하지 않습니다.")
        return
    
    # 삭제 확인
    response = input(f"❓ {len(to_delete)}개 체크포인트를 삭제하시겠습니까? (y/N): ")
    if response.lower() != 'y':
        print("❌ 취소되었습니다.")
        return
    
    # 삭제 실행
    deleted_count = 0
    for info in to_delete:
        try:
            os.remove(info['path'])
            deleted_count += 1
            print(f"✅ 삭제: {Path(info['path']).name}")
        except Exception as e:
            print(f"❌ 삭제 실패: {Path(info['path']).name} - {e}")
    
    print()
    print(f"✅ 완료: {deleted_count}개 삭제, {delete_size_gb:.2f} GB 확보")


def main():
    parser = argparse.ArgumentParser(description='체크포인트 자동 정리')
    parser.add_argument('--runs-dir', type=str, default='runs',
                        help='runs 디렉토리 경로 (기본: runs)')
    parser.add_argument('--keep', type=int, default=3,
                        help='유지할 최근 체크포인트 개수 (기본: 3)')
    parser.add_argument('--dry-run', action='store_true',
                        help='실제 삭제 없이 테스트')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("🧹 체크포인트 정리 스크립트")
    print("=" * 50)
    print()
    
    cleanup_checkpoints(args.runs_dir, keep_last=args.keep, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
