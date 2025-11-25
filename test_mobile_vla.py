#!/usr/bin/env python3
"""
Mobile VLA 데이터 수집 시스템 테스트 스크립트
- 시뮬레이션 모드로 데이터 수집 테스트
- Action chunk 생성 검증
- HDF5 저장/로드 테스트
"""

import sys
import time
import numpy as np
from pathlib import Path
import h5py
import cv2

# 프로젝트 경로 추가
sys.path.append(str(Path(__file__).parent))

def test_wasd_mapping():
    """WASD → 2D 연속값 매핑 테스트"""
    print("🧪 WASD 매핑 테스트")
    
    # ROS2 노드 없이 매핑만 테스트
    WASD_TO_CONTINUOUS = {
        'w': {"linear_x": 0.5, "linear_y": 0.0, "angular_z": 0.0},   # 전진
        'a': {"linear_x": 0.0, "linear_y": 0.5, "angular_z": 0.0},   # 좌이동  
        's': {"linear_x": -0.5, "linear_y": 0.0, "angular_z": 0.0},  # 후진
        'd': {"linear_x": 0.0, "linear_y": -0.5, "angular_z": 0.0},  # 우이동
        'q': {"linear_x": 0.5, "linear_y": 0.5, "angular_z": 0.0},   # 전좌대각
        'e': {"linear_x": 0.5, "linear_y": -0.5, "angular_z": 0.0},  # 전우대각
        'z': {"linear_x": -0.5, "linear_y": 0.5, "angular_z": 0.0},  # 후좌대각
        'c': {"linear_x": -0.5, "linear_y": -0.5, "angular_z": 0.0}, # 후우대각
        'r': {"linear_x": 0.0, "linear_y": 0.0, "angular_z": 0.5},   # 좌회전
        't': {"linear_x": 0.0, "linear_y": 0.0, "angular_z": -0.5},  # 우회전
        ' ': {"linear_x": 0.0, "linear_y": 0.0, "angular_z": 0.0}    # 정지
    }
    
    # 테스트 케이스
    test_cases = ['w', 'a', 's', 'd', 'q', 'e', 'z', 'c', 'r', 't', ' ']
    
    print("키 → 액션 매핑:")
    for key in test_cases:
        if key in WASD_TO_CONTINUOUS:
            action = WASD_TO_CONTINUOUS[key]
            print(f"  {key.upper():>2} → linear_x={action['linear_x']:+.1f}, "
                  f"linear_y={action['linear_y']:+.1f}, angular_z={action['angular_z']:+.1f}")
    
    print("✅ WASD 매핑 테스트 완료\n")
    return WASD_TO_CONTINUOUS

def test_action_chunk_creation():
    """Action Chunk 생성 테스트"""
    print("🧪 Action Chunk 생성 테스트")
    
    # ROS2 노드 없이 Action Chunk 로직만 테스트
    class MockDataCollector:
        def __init__(self):
            self.WINDOW_SIZE = 10      # 과거 프레임
            self.CHUNK_SIZE = 8        # 미래 프레임
            self.TOTAL_FRAMES = self.WINDOW_SIZE + self.CHUNK_SIZE  # 18프레임
            self.action_history = []
            self.image_history = []
            self.episode_data = {}
            
        def create_action_chunks(self):
            """RoboVLMs 방식의 Action Chunk 생성"""
            if len(self.action_history) < self.TOTAL_FRAMES:
                print(f"데이터 부족: {len(self.action_history)} < {self.TOTAL_FRAMES}")
                return
                
            chunks = []
            for i in range(len(self.action_history) - self.TOTAL_FRAMES + 1):
                # 과거 10프레임 + 미래 8프레임 추출
                chunk_actions = self.action_history[i:i+self.TOTAL_FRAMES]
                chunk_images = self.image_history[i:i+self.TOTAL_FRAMES] if len(self.image_history) >= self.TOTAL_FRAMES else []
                
                chunk = {
                    "chunk_id": len(chunks),
                    "timestamp": chunk_actions[self.WINDOW_SIZE]["timestamp"],  # 현재 시점
                    "past_actions": [a["action"] for a in chunk_actions[:self.WINDOW_SIZE]],  # 과거 10개
                    "future_actions": [a["action"] for a in chunk_actions[self.WINDOW_SIZE:]],  # 미래 8개
                    "images": [img["image"] for img in chunk_images] if chunk_images else [],
                    "window_size": self.WINDOW_SIZE,
                    "chunk_size": self.CHUNK_SIZE
                }
                chunks.append(chunk)
                
            self.episode_data["action_chunks"] = chunks
            print(f"📊 생성된 Action Chunks: {len(chunks)}개")
    
    collector = MockDataCollector()
    
    # 시뮬레이션 데이터 생성 (25개 액션)
    print("📊 시뮬레이션 데이터 생성 중...")
    
    for i in range(25):  # 18프레임 + 여유분
        # 간단한 시뮬레이션 액션 (앞으로 가다가 회전)
        if i < 10:
            action = {"linear_x": 0.5, "linear_y": 0.0, "angular_z": 0.0}
        elif i < 15:
            action = {"linear_x": 0.0, "linear_y": 0.0, "angular_z": 0.5}
        else:
            action = {"linear_x": 0.3, "linear_y": 0.2, "angular_z": 0.0}
            
        collector.action_history.append({
            "action": action,
            "timestamp": time.time() + i * 0.1
        })
        
        # 가짜 이미지 추가
        fake_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        collector.image_history.append({
            "image": fake_image,
            "timestamp": time.time() + i * 0.1
        })
    
    print(f"📦 생성된 액션 수: {len(collector.action_history)}")
    print(f"🖼️  생성된 이미지 수: {len(collector.image_history)}")
    
    # Action chunk 생성
    collector.episode_data = {
        "episode_name": "test_episode",
        "action_chunks": [],
        "total_duration": 2.5,
        "obstacle_config": {},
        "cup_position": {"x": 0.0, "y": 1.0}
    }
    
    collector.create_action_chunks()
    
    print(f"✅ 생성된 Action Chunks: {len(collector.episode_data['action_chunks'])}개")
    
    # 첫 번째 청크 분석
    if collector.episode_data['action_chunks']:
        first_chunk = collector.episode_data['action_chunks'][0]
        print(f"🔍 첫 번째 청크 분석:")
        print(f"   - Chunk ID: {first_chunk['chunk_id']}")
        print(f"   - 과거 액션 수: {len(first_chunk['past_actions'])}")
        print(f"   - 미래 액션 수: {len(first_chunk['future_actions'])}")
        print(f"   - 이미지 수: {len(first_chunk['images'])}")
        print(f"   - Window Size: {first_chunk['window_size']}")
        print(f"   - Chunk Size: {first_chunk['chunk_size']}")
    
    print("✅ Action Chunk 생성 테스트 완료\n")
    
    return collector

def test_hdf5_save_load():
    """HDF5 저장/로드 테스트"""
    print("🧪 HDF5 저장/로드 테스트")
    
    # 이전 테스트에서 데이터가 있는 collector 생성
    collector = test_action_chunk_creation()
    
    # HDF5 저장 함수 (ROS2 노드 없이)
    def save_episode_data_mock(collector, data_dir):
        data_dir = Path(data_dir)
        data_dir.mkdir(exist_ok=True)
        
        episode_name = collector.episode_data["episode_name"]
        save_path = data_dir / f"{episode_name}.h5"
        
        with h5py.File(save_path, 'w') as f:
            # 메타데이터
            f.attrs['episode_name'] = episode_name
            f.attrs['total_duration'] = collector.episode_data["total_duration"]
            f.attrs['cup_position_x'] = collector.episode_data["cup_position"]["x"]
            f.attrs['cup_position_y'] = collector.episode_data["cup_position"]["y"]
            f.attrs['window_size'] = collector.WINDOW_SIZE
            f.attrs['chunk_size'] = collector.CHUNK_SIZE
            
            # Action chunks
            chunks_group = f.create_group('action_chunks')
            for i, chunk in enumerate(collector.episode_data["action_chunks"]):
                chunk_group = chunks_group.create_group(f'chunk_{i}')
                chunk_group.attrs['chunk_id'] = chunk["chunk_id"]
                chunk_group.attrs['timestamp'] = chunk["timestamp"]
                
                # 과거/미래 액션 저장
                past_actions = np.array([[a["linear_x"], a["linear_y"], a["angular_z"]] 
                                       for a in chunk["past_actions"]])
                future_actions = np.array([[a["linear_x"], a["linear_y"], a["angular_z"]] 
                                         for a in chunk["future_actions"]])
                
                chunk_group.create_dataset('past_actions', data=past_actions)
                chunk_group.create_dataset('future_actions', data=future_actions)
                
                # 이미지 저장 (있는 경우)
                if chunk["images"]:
                    images = np.stack(chunk["images"])  # [frames, height, width, channels]
                    chunk_group.create_dataset('images', data=images, compression='gzip')
        
        return save_path
    
    # 임시 저장 경로
    test_dir = Path("test_mobile_vla_data")
    
    # 저장 테스트
    print("💾 HDF5 파일 저장 중...")
    save_path = save_episode_data_mock(collector, test_dir)
    print(f"✅ 저장 완료: {save_path}")
    
    # 로드 테스트
    print("📂 HDF5 파일 로드 중...")
    with h5py.File(save_path, 'r') as f:
        print("🔍 파일 구조 분석:")
        print(f"   - Episode Name: {f.attrs.get('episode_name', 'N/A')}")
        print(f"   - Total Duration: {f.attrs.get('total_duration', 0.0):.2f}초")
        print(f"   - Cup Position: ({f.attrs.get('cup_position_x', 0.0)}, {f.attrs.get('cup_position_y', 0.0)})")
        print(f"   - Window Size: {f.attrs.get('window_size', 0)}")
        print(f"   - Chunk Size: {f.attrs.get('chunk_size', 0)}")
        
        if 'action_chunks' in f:
            chunks_group = f['action_chunks']
            print(f"   - Action Chunks: {len(chunks_group.keys())}개")
            
            # 첫 번째 청크 상세 분석
            if 'chunk_0' in chunks_group:
                chunk_0 = chunks_group['chunk_0']
                print(f"   - 첫 번째 청크:")
                print(f"     * Chunk ID: {chunk_0.attrs.get('chunk_id', -1)}")
                print(f"     * Timestamp: {chunk_0.attrs.get('timestamp', 0.0):.3f}")
                
                if 'past_actions' in chunk_0:
                    past_actions = chunk_0['past_actions'][:]
                    print(f"     * Past Actions Shape: {past_actions.shape}")
                    print(f"     * Past Actions Sample: {past_actions[0]}")
                
                if 'future_actions' in chunk_0:
                    future_actions = chunk_0['future_actions'][:]
                    print(f"     * Future Actions Shape: {future_actions.shape}")
                    print(f"     * Future Actions Sample: {future_actions[0]}")
                
                if 'images' in chunk_0:
                    images = chunk_0['images']
                    print(f"     * Images Shape: {images.shape}")
    
    print("✅ HDF5 저장/로드 테스트 완료\n")
    
    return save_path

def test_data_inspector():
    """데이터 검사 도구 테스트"""
    print("🧪 데이터 검사 도구 테스트")
    
    # 테스트 데이터 생성
    test_file = test_hdf5_save_load()
    
    # 간단한 검사 함수 (데이터 검사 도구 기능 시뮬레이션)
    def inspect_episode_mock(episode_path):
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
                
                # 첫 번째 청크만 분석
                if 'chunk_0' in chunks_group:
                    chunk = chunks_group['chunk_0']
                    
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
                        
                    if 'future_actions' in chunk:
                        future_actions = chunk['future_actions'][:]
                        chunk_detail['future_actions_shape'] = future_actions.shape
                        chunk_detail['future_actions_mean'] = np.mean(future_actions, axis=0).tolist()
                    
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
    
    # 검사 실행
    episode_info = inspect_episode_mock(test_file)
    
    print("🔍 검사 결과:")
    print(f"📁 파일: {episode_info['file_path'].name}")
    print(f"💾 크기: {episode_info['file_size_mb']:.2f} MB")
    
    # 메타데이터
    metadata = episode_info['metadata']
    print(f"🎬 에피소드명: {metadata['episode_name']}")
    print(f"⏱️  총 시간: {metadata['total_duration']:.2f}초")
    print(f"🎯 컵 위치: ({metadata['cup_position']['x']:.1f}, {metadata['cup_position']['y']:.1f})")
    print(f"📊 프레임 구조: 과거 {metadata['window_size']}개 + 미래 {metadata['chunk_size']}개")
    
    # Action chunks
    chunks_info = episode_info['chunks_info']
    print(f"📦 총 Action Chunks: {chunks_info['total_chunks']}개")
    
    # 액션 패턴 분석
    if chunks_info['chunk_details']:
        first_chunk = chunks_info['chunk_details'][0]
        print(f"🔍 첫 번째 청크:")
        print(f"   - Past Actions: {first_chunk.get('past_actions_shape', 'N/A')}")
        print(f"   - Future Actions: {first_chunk.get('future_actions_shape', 'N/A')}")
        print(f"   - 이미지 포함: {'✅' if first_chunk.get('has_images', False) else '❌'}")
        
        if 'past_actions_mean' in first_chunk:
            past_mean = first_chunk['past_actions_mean']
            print(f"   - 과거 액션 평균: linear_x={past_mean[0]:.3f}, linear_y={past_mean[1]:.3f}, angular_z={past_mean[2]:.3f}")
        
        if 'future_actions_mean' in first_chunk:
            future_mean = first_chunk['future_actions_mean']
            print(f"   - 미래 액션 평균: linear_x={future_mean[0]:.3f}, linear_y={future_mean[1]:.3f}, angular_z={future_mean[2]:.3f}")
    
    print("✅ 데이터 검사 도구 테스트 완료\n")

def cleanup_test_data():
    """테스트 데이터 정리"""
    print("🧹 테스트 데이터 정리 중...")
    
    test_dir = Path("test_mobile_vla_data")
    if test_dir.exists():
        for file in test_dir.glob("*.h5"):
            file.unlink()
        test_dir.rmdir()
        print("✅ 테스트 데이터 정리 완료")

def main():
    """전체 테스트 실행"""
    print("🚀 Mobile VLA 데이터 수집 시스템 테스트 시작")
    print("=" * 60)
    
    try:
        # 1. WASD 매핑 테스트
        test_wasd_mapping()
        
        # 2. Action Chunk 생성 테스트  
        test_action_chunk_creation()
        
        # 3. HDF5 저장/로드 테스트
        test_hdf5_save_load()
        
        # 4. 데이터 검사 도구 테스트
        test_data_inspector()
        
        print("🎉 모든 테스트 완료!")
        print("=" * 60)
        print("📋 다음 단계:")
        print("   1. 실제 로봇 환경에서 데이터 수집 테스트")
        print("   2. 컵 추적 시나리오 구현")
        print("   3. 장애물 배치 시나리오 설계")
        print("   4. 서버 학습 파이프라인 연동")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        cleanup_test_data()

if __name__ == '__main__':
    main()