#!/usr/bin/env python3
"""
Mobile VLA Dataset Classifier
프레임 수 기반으로 데이터셋을 자동 분류하고 태그를 붙이는 유틸리티
"""

import h5py
import json
import shutil
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from datetime import datetime

class DatasetClassifier:
    def __init__(self, dataset_path: str = "mobile_vla_dataset"):
        """
        Args:
            dataset_path: 데이터셋이 있는 경로
        """
        self.dataset_path = Path(dataset_path)
        self.categories = {
            "short": {"min": 1, "max": 10, "target": 50, "description": "짧은 에피소드 (1-10 프레임)"},
            "medium": {"min": 11, "max": 25, "target": 100, "description": "중간 에피소드 (11-25 프레임)"},
            "long": {"min": 26, "max": 50, "target": 30, "description": "긴 에피소드 (26-50 프레임)"},
            "extra_long": {"min": 51, "max": float('inf'), "target": 10, "description": "매우 긴 에피소드 (51+ 프레임)"}
        }
        
        # 분류된 데이터를 저장할 디렉토리들
        self.create_category_directories()
        
    def create_category_directories(self):
        """카테고리별 디렉토리 생성"""
        for category in self.categories.keys():
            category_dir = self.dataset_path / f"classified_{category}"
            category_dir.mkdir(exist_ok=True)
            
    def analyze_h5_file(self, h5_file: Path) -> Dict:
        """H5 파일을 분석하여 메타데이터 추출"""
        try:
            with h5py.File(h5_file, 'r') as f:
                metadata = {
                    "file_path": str(h5_file),
                    "episode_name": f.attrs.get('episode_name', h5_file.stem),
                    "total_duration": f.attrs.get('total_duration', 0.0),
                    "num_frames": f.attrs.get('num_frames', 0),
                    "action_chunk_size": f.attrs.get('action_chunk_size', 0),
                    "images_shape": f['images'].shape if 'images' in f else None,
                    "actions_shape": f['actions'].shape if 'actions' in f else None,
                    "file_size_mb": h5_file.stat().st_size / (1024 * 1024)
                }
                
                # 실제 프레임 수 확인 (images 데이터셋 기준)
                if 'images' in f:
                    metadata["actual_frames"] = f['images'].shape[0]
                else:
                    metadata["actual_frames"] = 0
                    
                return metadata
        except Exception as e:
            print(f"❌ {h5_file} 분석 실패: {e}")
            return None
            
    def classify_by_frames(self, num_frames: int) -> str:
        """프레임 수에 따라 카테고리 분류"""
        for category, config in self.categories.items():
            if config["min"] <= num_frames <= config["max"]:
                return category
        return "unknown"
        
    def scan_dataset(self) -> Dict:
        """전체 데이터셋을 스캔하여 분류"""
        print("🔍 데이터셋 스캔 중...")
        
        h5_files = list(self.dataset_path.glob("*.h5"))
        results = {
            "total_files": len(h5_files),
            "classified": defaultdict(list),
            "statistics": defaultdict(int),
            "errors": []
        }
        
        for h5_file in h5_files:
            metadata = self.analyze_h5_file(h5_file)
            
            if metadata is None:
                results["errors"].append(str(h5_file))
                continue
                
            # 프레임 수 기준 분류
            category = self.classify_by_frames(metadata["actual_frames"])
            metadata["category"] = category
            
            results["classified"][category].append(metadata)
            results["statistics"][category] += 1
            
        return results
        
    def generate_report(self, scan_results: Dict) -> str:
        """스캔 결과 리포트 생성"""
        report = []
        report.append("=" * 60)
        report.append("🤖 Mobile VLA Dataset Classification Report")
        report.append("=" * 60)
        report.append(f"📊 총 파일 수: {scan_results['total_files']}")
        report.append(f"❌ 에러 파일 수: {len(scan_results['errors'])}")
        report.append("")
        
        # 카테고리별 통계
        report.append("📈 카테고리별 현황:")
        report.append("-" * 60)
        
        total_current = 0
        total_target = 0
        
        for category, config in self.categories.items():
            current = scan_results["statistics"][category]
            target = config["target"]
            percentage = (current / target * 100) if target > 0 else 0
            
            total_current += current
            total_target += target
            
            status_emoji = "✅" if current >= target else "⏳"
            progress_bar = self.create_progress_bar(current, target)
            
            report.append(f"{status_emoji} {category.upper()}: {current}/{target}개 ({percentage:.1f}%)")
            report.append(f"   {config['description']}")
            report.append(f"   {progress_bar}")
            report.append("")
            
        # 전체 진행률
        overall_percentage = (total_current / total_target * 100) if total_target > 0 else 0
        overall_progress = self.create_progress_bar(total_current, total_target, width=40)
        
        report.append("🎯 전체 진행률:")
        report.append(f"   {total_current}/{total_target}개 ({overall_percentage:.1f}%)")
        report.append(f"   {overall_progress}")
        report.append("")
        
        # 프레임 18개 데이터 강조
        frame_18_files = []
        for category_files in scan_results["classified"].values():
            for metadata in category_files:
                if metadata["actual_frames"] == 18:
                    frame_18_files.append(metadata)
                    
        if frame_18_files:
            report.append("🎯 프레임 18개 데이터 (특별 관심 대상):")
            report.append("-" * 40)
            for metadata in frame_18_files:
                report.append(f"   • {metadata['episode_name']} (카테고리: {metadata['category']})")
            report.append("")
            
        # 에러 파일 리스트
        if scan_results["errors"]:
            report.append("❌ 에러 파일들:")
            for error_file in scan_results["errors"]:
                report.append(f"   • {error_file}")
            report.append("")
            
        report.append("=" * 60)
        
        return "\n".join(report)
        
    def create_progress_bar(self, current: int, target: int, width: int = 20) -> str:
        """진행률 바 생성"""
        if target == 0:
            return "█" * width + " (무제한)"
            
        percentage = min(current / target, 1.0)
        filled = int(width * percentage)
        bar = "█" * filled + "░" * (width - filled)
        return f"{bar} {current}/{target}"
        
    def organize_files(self, scan_results: Dict, copy_mode: bool = True):
        """파일들을 카테고리별로 정리 (복사 또는 이동)"""
        print(f"📁 파일 정리 중... ({'복사' if copy_mode else '이동'} 모드)")
        
        for category, files in scan_results["classified"].items():
            category_dir = self.dataset_path / f"classified_{category}"
            category_dir.mkdir(exist_ok=True)
            
            for metadata in files:
                source_path = Path(metadata["file_path"])
                dest_path = category_dir / source_path.name
                
                try:
                    if copy_mode:
                        shutil.copy2(source_path, dest_path)
                        print(f"📋 복사: {source_path.name} → {category}/")
                    else:
                        shutil.move(str(source_path), str(dest_path))
                        print(f"📁 이동: {source_path.name} → {category}/")
                except Exception as e:
                    print(f"❌ 파일 처리 실패 {source_path.name}: {e}")
                    
    def save_metadata(self, scan_results: Dict):
        """분석 결과를 JSON 파일로 저장"""
        metadata_file = self.dataset_path / "dataset_classification_metadata.json"
        
        # JSON 직렬화 가능하도록 데이터 변환
        json_data = {
            "timestamp": datetime.now().isoformat(),
            "total_files": scan_results["total_files"],
            "statistics": dict(scan_results["statistics"]),
            "categories": self.categories,
            "files": {}
        }
        
        for category, files in scan_results["classified"].items():
            json_data["files"][category] = [
                {k: int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v 
                 for k, v in metadata.items() if k != "file_path"}
                for metadata in files
            ]
            
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
            
        print(f"💾 메타데이터 저장: {metadata_file}")
        
    def create_tagged_symlinks(self, scan_results: Dict):
        """태그된 심볼릭 링크 생성"""
        tagged_dir = self.dataset_path / "tagged_episodes"
        tagged_dir.mkdir(exist_ok=True)
        
        print("🏷️ 태그된 심볼릭 링크 생성 중...")
        
        for category, files in scan_results["classified"].items():
            for metadata in files:
                source_path = Path(metadata["file_path"])
                frames = metadata["actual_frames"]
                
                # 태그된 파일명 생성
                tagged_name = f"{category}_{frames}frames_{source_path.name}"
                tagged_path = tagged_dir / tagged_name
                
                try:
                    # 기존 심볼릭 링크가 있으면 제거
                    if tagged_path.is_symlink():
                        tagged_path.unlink()
                        
                    # 상대 경로로 심볼릭 링크 생성
                    relative_source = Path("..") / source_path.name
                    tagged_path.symlink_to(relative_source)
                    
                    # 프레임 18개 데이터는 특별 표시
                    if frames == 18:
                        special_name = f"★FRAME18★_{tagged_name}"
                        special_path = tagged_dir / special_name
                        if special_path.is_symlink():
                            special_path.unlink()
                        special_path.symlink_to(relative_source)
                        
                except Exception as e:
                    print(f"❌ 심볼릭 링크 생성 실패 {tagged_name}: {e}")

def main():
    """메인 실행 함수"""
    print("🤖 Mobile VLA Dataset Classifier 시작!")
    
    # 데이터셋 경로 설정 (현재 디렉토리 기준)
    dataset_path = "mobile_vla_dataset"
    
    if not Path(dataset_path).exists():
        print(f"❌ 데이터셋 경로를 찾을 수 없습니다: {dataset_path}")
        return
        
    classifier = DatasetClassifier(dataset_path)
    
    # 1. 데이터셋 스캔
    scan_results = classifier.scan_dataset()
    
    # 2. 리포트 생성 및 출력
    report = classifier.generate_report(scan_results)
    print(report)
    
    # 3. 메타데이터 저장
    classifier.save_metadata(scan_results)
    
    # 4. 태그된 심볼릭 링크 생성
    classifier.create_tagged_symlinks(scan_results)
    
    # 5. 사용자 선택: 파일 정리
    print("\n📁 파일 정리 옵션:")
    print("1. 복사 모드 (원본 파일 유지)")
    print("2. 이동 모드 (원본 파일 이동)")
    print("3. 건너뛰기")
    
    choice = input("선택하세요 (1/2/3): ").strip()
    
    if choice == "1":
        classifier.organize_files(scan_results, copy_mode=True)
    elif choice == "2":
        classifier.organize_files(scan_results, copy_mode=False)
    else:
        print("파일 정리를 건너뜁니다.")
        
    print("\n✅ 분류 작업 완료!")
    print(f"📂 태그된 파일들: {dataset_path}/tagged_episodes/")
    print(f"📋 메타데이터: {dataset_path}/dataset_classification_metadata.json")

if __name__ == "__main__":
    main()
