#!/usr/bin/env python3
"""
체크포인트 파일 손상 여부 정확한 진단
"""

import torch
import os
import json
import hashlib
from typing import Dict, Any, List

class CheckpointDamageAnalyzer:
    """체크포인트 손상 분석기"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Device: {self.device}")
    
    def calculate_file_hash(self, file_path: str) -> str:
        """파일 해시 계산"""
        try:
            with open(file_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            return file_hash
        except Exception as e:
            print(f"❌ Hash calculation failed: {e}")
            return "ERROR"
    
    def check_file_integrity(self, file_path: str) -> Dict[str, Any]:
        """파일 무결성 검사"""
        print(f"\n🔍 Checking file integrity: {file_path}")
        print("-" * 60)
        
        result = {
            'file_path': file_path,
            'exists': False,
            'file_size_mb': 0,
            'file_size_bytes': 0,
            'file_hash': '',
            'can_read': False,
            'torch_loadable': False,
            'checkpoint_structure': {},
            'error_message': '',
            'damage_assessment': 'Unknown'
        }
        
        # 1. 파일 존재 여부
        if not os.path.exists(file_path):
            result['error_message'] = "File does not exist"
            result['damage_assessment'] = "File Missing"
            return result
        
        result['exists'] = True
        
        # 2. 파일 크기 확인
        try:
            file_size = os.path.getsize(file_path)
            result['file_size_bytes'] = file_size
            result['file_size_mb'] = file_size / (1024 * 1024)
            print(f"📏 File size: {result['file_size_mb']:.1f}MB ({file_size:,} bytes)")
        except Exception as e:
            result['error_message'] = f"Size check failed: {e}"
            result['damage_assessment'] = "Size Check Failed"
            return result
        
        # 3. 파일 해시 계산
        try:
            result['file_hash'] = self.calculate_file_hash(file_path)
            print(f"🔐 File hash: {result['file_hash']}")
        except Exception as e:
            result['error_message'] = f"Hash calculation failed: {e}"
            result['damage_assessment'] = "Hash Calculation Failed"
            return result
        
        # 4. 파일 읽기 가능 여부
        try:
            with open(file_path, 'rb') as f:
                # 파일의 처음 1024바이트 읽기
                header = f.read(1024)
                if len(header) > 0:
                    result['can_read'] = True
                    print(f"✅ File is readable")
                else:
                    result['error_message'] = "File is empty"
                    result['damage_assessment'] = "Empty File"
                    return result
        except Exception as e:
            result['error_message'] = f"File read failed: {e}"
            result['damage_assessment'] = "Read Failed"
            return result
        
        # 5. PyTorch 로드 가능 여부
        try:
            print(f"🔄 Attempting to load with PyTorch...")
            checkpoint = torch.load(file_path, map_location='cpu')
            result['torch_loadable'] = True
            print(f"✅ PyTorch load successful")
            
            # 체크포인트 구조 분석
            if isinstance(checkpoint, dict):
                result['checkpoint_structure'] = {
                    'keys': list(checkpoint.keys()),
                    'has_model_state_dict': 'model_state_dict' in checkpoint,
                    'has_optimizer_state_dict': 'optimizer_state_dict' in checkpoint,
                    'has_epoch': 'epoch' in checkpoint,
                    'has_val_mae': 'val_mae' in checkpoint,
                    'model_state_dict_keys': []
                }
                
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    if isinstance(state_dict, dict):
                        result['checkpoint_structure']['model_state_dict_keys'] = list(state_dict.keys())
                        result['checkpoint_structure']['parameter_count'] = sum(p.numel() for p in state_dict.values())
                
                print(f"📊 Checkpoint structure: {result['checkpoint_structure']['keys']}")
                
            else:
                result['checkpoint_structure'] = {
                    'type': type(checkpoint).__name__,
                    'message': 'Not a dictionary checkpoint'
                }
            
            result['damage_assessment'] = "Healthy"
            
        except Exception as e:
            result['error_message'] = f"PyTorch load failed: {e}"
            result['damage_assessment'] = "PyTorch Load Failed"
            print(f"❌ PyTorch load failed: {e}")
        
        return result
    
    def analyze_all_checkpoints(self) -> Dict[str, Any]:
        """모든 체크포인트 분석"""
        print("🚀 Starting comprehensive checkpoint damage analysis")
        print("=" * 80)
        
        checkpoints = [
            "Robo+/Mobile_VLA/results/simple_clip_lstm_results_extended/best_simple_clip_lstm_model.pth",
            "Robo+/Mobile_VLA/results/simple_lstm_results_extended/best_simple_lstm_model.pth",
            "Robo+/Mobile_VLA/results/simple_lstm_results_extended/final_simple_lstm_model.pth"
        ]
        
        results = {}
        for checkpoint_path in checkpoints:
            model_name = os.path.basename(checkpoint_path).replace('.pth', '')
            results[model_name] = self.check_file_integrity(checkpoint_path)
        
        return results
    
    def create_damage_report(self, results: Dict[str, Any]):
        """손상 보고서 생성"""
        print(f"\n📋 DAMAGE ASSESSMENT REPORT")
        print("=" * 80)
        
        # 상태별 분류
        status_counts = {}
        for model_name, result in results.items():
            status = result['damage_assessment']
            if status not in status_counts:
                status_counts[status] = []
            status_counts[status].append(model_name)
        
        print(f"\n📊 **Status Summary**:")
        for status, models in status_counts.items():
            print(f"   - {status}: {len(models)} models")
            for model in models:
                print(f"     * {model}")
        
        print(f"\n🔍 **Detailed Analysis**:")
        for model_name, result in results.items():
            print(f"\n🏷️  **{model_name}**:")
            print(f"   - Status: {result['damage_assessment']}")
            print(f"   - File size: {result['file_size_mb']:.1f}MB")
            print(f"   - File hash: {result['file_hash']}")
            print(f"   - Readable: {result['can_read']}")
            print(f"   - PyTorch loadable: {result['torch_loadable']}")
            
            if result['error_message']:
                print(f"   - Error: {result['error_message']}")
            
            if result['checkpoint_structure']:
                structure = result['checkpoint_structure']
                if 'parameter_count' in structure:
                    print(f"   - Parameters: {structure['parameter_count']:,}")
                if 'model_state_dict_keys' in structure:
                    print(f"   - State dict keys: {len(structure['model_state_dict_keys'])}")
        
        # 손상 원인 추정
        print(f"\n🔍 **Damage Cause Analysis**:")
        print("-" * 60)
        
        for model_name, result in results.items():
            if result['damage_assessment'] != "Healthy":
                print(f"\n❌ **{model_name} - Possible causes**:")
                
                if result['damage_assessment'] == "File Missing":
                    print("   - File was deleted or moved")
                    print("   - Path is incorrect")
                    print("   - Permission issues")
                
                elif result['damage_assessment'] == "Size Check Failed":
                    print("   - File system corruption")
                    print("   - Disk space issues")
                    print("   - Permission problems")
                
                elif result['damage_assessment'] == "Read Failed":
                    print("   - File corruption")
                    print("   - Permission denied")
                    print("   - File system issues")
                
                elif result['damage_assessment'] == "PyTorch Load Failed":
                    print("   - Incomplete file download")
                    print("   - File corruption during transfer")
                    print("   - PyTorch version incompatibility")
                    print("   - Memory issues during loading")
                
                elif result['damage_assessment'] == "Empty File":
                    print("   - Incomplete file creation")
                    print("   - Interrupted file transfer")
                    print("   - Disk space issues during save")
        
        # 해결 방안 제시
        print(f"\n💡 **Recommended Solutions**:")
        print("-" * 60)
        
        healthy_count = len([r for r in results.values() if r['damage_assessment'] == "Healthy"])
        total_count = len(results)
        
        if healthy_count == total_count:
            print("✅ All checkpoints are healthy!")
            print("   - No damage detected")
            print("   - Files are ready for use")
            print("   - Issue might be in loading code or environment")
        else:
            print(f"⚠️  {total_count - healthy_count} out of {total_count} checkpoints have issues")
            print("   - Check file system integrity")
            print("   - Verify file permissions")
            print("   - Re-download corrupted files")
            print("   - Check PyTorch version compatibility")
    
    def save_analysis_results(self, results: Dict[str, Any]):
        """분석 결과 저장"""
        output_path = "Robo+/Mobile_VLA/checkpoint_damage_analysis_results.json"
        
        with open(output_path, "w") as f:
            json.dump({
                'analysis_results': results,
                'timestamp': '2024-08-22',
                'summary': {
                    'total_checkpoints': len(results),
                    'healthy_count': len([r for r in results.values() if r['damage_assessment'] == "Healthy"]),
                    'damaged_count': len([r for r in results.values() if r['damage_assessment'] != "Healthy"])
                }
            }, f, indent=2)
        
        print(f"\n✅ Analysis results saved to: {output_path}")

def main():
    """메인 함수"""
    print("🚀 Starting Checkpoint Damage Analysis")
    print("🎯 Determining why files were considered damaged")
    
    analyzer = CheckpointDamageAnalyzer()
    
    try:
        # 모든 체크포인트 분석
        results = analyzer.analyze_all_checkpoints()
        
        # 손상 보고서 생성
        analyzer.create_damage_report(results)
        
        # 결과 저장
        analyzer.save_analysis_results(results)
        
        print(f"\n✅ Checkpoint damage analysis completed!")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        raise

if __name__ == "__main__":
    main()
