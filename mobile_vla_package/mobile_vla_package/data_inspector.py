#!/usr/bin/env python3
"""
Mobile VLA 데이터셋 검사 도구
- HDF5 파일 구조 분석
- Action chunk 통계 출력
- 데이터 무결성 검증
"""

import h5py
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List
import cv2

class MobileVLADataInspector:
    """Mobile VLA 데이터셋 검사 클래스"""
    
    def __init__(self, data_path: Path):
        self.data_path = Path(data_path)
        self.episodes = []
        
        if self.data_path.is_file() and self.data_path.suffix == '.h5':
            self.episodes = [self.data_path]
        elif self.data_path.is_dir():
            self.episodes = list(self.data_path.glob("*.h5"))
        else:
            raise ValueError(f"Invalid path: {data_path}")
            
        print(f"📁 발견된 에피소드: {len(self.episodes)}개")

    def inspect_episode(self, episode_path: Path) -> Dict:
        """단일 에피소드 검사"""
        with h5py.File(episode_path, 'r') as f:
            # 메타데이터
            metadata = {
                'episode_name': f.attrs.get('episode_name', 'Unknown'),
                'total_duration': f.attrs.get('total_duration', 0.0),
                'cup_position': {
                    'x': f.attrs.get('cup_position_x', 0.0),
                    'y': f.attrs.get('cup_position_y', 0.0)
                },
                'window_size': f.attrs.get('window_size', 10),
                'chunk_size': f.attrs.get('chunk_size', 8)
            }
            
            # Action chunks 분석
            chunks_info = {
                'total_chunks': 0,
                'chunk_details': []
            }
            
            if 'action_chunks' in f:
                chunks_group = f['action_chunks']
                chunks_info['total_chunks'] = len(chunks_group.keys())
                
                for chunk_key in chunks_group.keys():
                    chunk = chunks_group[chunk_key]
                    
                    chunk_detail = {
                        'chunk_id': chunk.attrs.get('chunk_id', -1),
                        'timestamp': chunk.attrs.get('timestamp', 0.0),
                        'has_images': 'images' in chunk,
                    }
                    
                    # 액션 분석
                    if 'past_actions' in chunk:
                        past_actions = chunk['past_actions'][:]
                        chunk_detail['past_actions_shape'] = past_actions.shape
                        chunk_detail['past_actions_mean'] = np.mean(past_actions, axis=0).tolist()
                        chunk_detail['past_actions_std'] = np.std(past_actions, axis=0).tolist()
                        
                    if 'future_actions' in chunk:
                        future_actions = chunk['future_actions'][:]
                        chunk_detail['future_actions_shape'] = future_actions.shape
                        chunk_detail['future_actions_mean'] = np.mean(future_actions, axis=0).tolist()
                        chunk_detail['future_actions_std'] = np.std(future_actions, axis=0).tolist()
                    
                    # 이미지 분석
                    if 'images' in chunk:
                        images = chunk['images']
                        chunk_detail['images_shape'] = images.shape
                        
                    chunks_info['chunk_details'].append(chunk_detail)
            
            return {
                'file_path': episode_path,
                'file_size_mb': episode_path.stat().st_size / (1024*1024),
                'metadata': metadata,
                'chunks_info': chunks_info
            }

    def print_episode_summary(self, episode_info: Dict):
        """에피소드 요약 출력"""
        print(f"\n{'='*60}")
        print(f"📁 파일: {episode_info['file_path'].name}")
        print(f"💾 크기: {episode_info['file_size_mb']:.2f} MB")
        print(f"{'='*60}")
        
        # 메타데이터
        metadata = episode_info['metadata']
        print(f"🎬 에피소드명: {metadata['episode_name']}")
        print(f"⏱️  총 시간: {metadata['total_duration']:.2f}초")
        print(f"🎯 컵 위치: ({metadata['cup_position']['x']:.1f}, {metadata['cup_position']['y']:.1f})")
        print(f"📊 프레임 구조: 과거 {metadata['window_size']}개 + 미래 {metadata['chunk_size']}개")
        
        # Action chunks
        chunks_info = episode_info['chunks_info']
        print(f"📦 총 Action Chunks: {chunks_info['total_chunks']}개")
        
        if chunks_info['chunk_details']:
            # 첫 번째 청크 상세 정보
            first_chunk = chunks_info['chunk_details'][0]
            print(f"🔍 첫 번째 청크:")
            print(f"   - Past Actions: {first_chunk.get('past_actions_shape', 'N/A')}")
            print(f"   - Future Actions: {first_chunk.get('future_actions_shape', 'N/A')}")
            print(f"   - 이미지 포함: {'✅' if first_chunk.get('has_images', False) else '❌'}")
            
            if first_chunk.get('has_images', False):
                img_shape = first_chunk.get('images_shape', (0, 0, 0, 0))
                print(f"   - 이미지 크기: {img_shape}")

    def analyze_action_patterns(self, episode_info: Dict):
        """액션 패턴 분석"""
        chunks_info = episode_info['chunks_info']
        
        if not chunks_info['chunk_details']:
            print("❌ 분석할 청크 데이터가 없습니다.")
            return
            
        print(f"\n📈 액션 패턴 분석:")
        
        # 모든 청크의 액션 통계
        all_past_means = []
        all_future_means = []
        
        for chunk in chunks_info['chunk_details']:
            if 'past_actions_mean' in chunk:
                all_past_means.append(chunk['past_actions_mean'])
            if 'future_actions_mean' in chunk:
                all_future_means.append(chunk['future_actions_mean'])
        
        if all_past_means:
            past_means = np.array(all_past_means)
            print(f"   과거 액션 평균: linear_x={past_means[:, 0].mean():.3f}, "
                  f"linear_y={past_means[:, 1].mean():.3f}, angular_z={past_means[:, 2].mean():.3f}")
                  
        if all_future_means:
            future_means = np.array(all_future_means)
            print(f"   미래 액션 평균: linear_x={future_means[:, 0].mean():.3f}, "
                  f"linear_y={future_means[:, 1].mean():.3f}, angular_z={future_means[:, 2].mean():.3f}")

    def visualize_first_chunk_images(self, episode_path: Path, output_dir: Path = None):
        """첫 번째 청크의 이미지들 시각화"""
        with h5py.File(episode_path, 'r') as f:
            if 'action_chunks' not in f:
                print("❌ Action chunks가 없습니다.")
                return
                
            chunks_group = f['action_chunks']
            if 'chunk_0' not in chunks_group:
                print("❌ 첫 번째 청크가 없습니다.")
                return
                
            chunk_0 = chunks_group['chunk_0']
            if 'images' not in chunk_0:
                print("❌ 첫 번째 청크에 이미지가 없습니다.")
                return
                
            images = chunk_0['images'][:]  # [frames, height, width, channels]
            
            print(f"🖼️  첫 번째 청크 이미지 시각화: {images.shape}")
            
            # 이미지 표시 (처음 8개만)
            num_show = min(8, images.shape[0])
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            axes = axes.flatten()
            
            for i in range(num_show):
                img = images[i]
                if img.shape[2] == 3:  # BGR to RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                axes[i].imshow(img)
                axes[i].set_title(f'Frame {i}')
                axes[i].axis('off')
            
            # 사용하지 않는 subplot 숨기기
            for i in range(num_show, len(axes)):
                axes[i].axis('off')
            
            plt.tight_layout()
            
            # 저장
            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(exist_ok=True)
                save_path = output_dir / f"{episode_path.stem}_chunk0_images.png"
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"💾 이미지 저장: {save_path}")
            
            plt.show()

    def inspect_all(self, show_details: bool = True, visualize: bool = False):
        """모든 에피소드 검사"""
        print(f"🔍 Mobile VLA 데이터셋 검사 시작")
        print(f"📂 경로: {self.data_path}")
        
        total_chunks = 0
        total_duration = 0.0
        total_size_mb = 0.0
        
        for episode_path in self.episodes:
            try:
                episode_info = self.inspect_episode(episode_path)
                
                if show_details:
                    self.print_episode_summary(episode_info)
                    self.analyze_action_patterns(episode_info)
                
                # 통계 누적
                total_chunks += episode_info['chunks_info']['total_chunks']
                total_duration += episode_info['metadata']['total_duration']
                total_size_mb += episode_info['file_size_mb']
                
                # 시각화 (첫 번째 에피소드만)
                if visualize and episode_path == self.episodes[0]:
                    self.visualize_first_chunk_images(episode_path)
                    
            except Exception as e:
                print(f"❌ 에피소드 {episode_path.name} 검사 실패: {e}")
        
        # 전체 통계
        print(f"\n{'='*60}")
        print(f"📊 전체 데이터셋 통계")
        print(f"{'='*60}")
        print(f"🎬 총 에피소드: {len(self.episodes)}개")
        print(f"📦 총 Action Chunks: {total_chunks}개")
        print(f"⏱️  총 수집 시간: {total_duration:.1f}초 ({total_duration/60:.1f}분)")
        print(f"💾 총 파일 크기: {total_size_mb:.2f} MB")
        if len(self.episodes) > 0:
            print(f"📊 평균 에피소드 크기: {total_size_mb/len(self.episodes):.2f} MB")
            print(f"📊 평균 청크 수: {total_chunks/len(self.episodes):.1f}개/에피소드")

def main():
    parser = argparse.ArgumentParser(description='Mobile VLA 데이터셋 검사 도구')
    parser.add_argument('data_path', type=str, help='HDF5 파일 또는 데이터셋 폴더 경로')
    parser.add_argument('--no-details', action='store_true', help='상세 정보 생략')
    parser.add_argument('--visualize', action='store_true', help='이미지 시각화')
    
    args = parser.parse_args()
    
    try:
        inspector = MobileVLADataInspector(args.data_path)
        inspector.inspect_all(
            show_details=not args.no_details,
            visualize=args.visualize
        )
    except Exception as e:
        print(f"❌ 검사 실패: {e}")

if __name__ == '__main__':
    main()