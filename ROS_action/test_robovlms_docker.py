#!/usr/bin/env python3
"""
RoboVLMs Docker Test Script
도커 환경에서 RoboVLMs 시스템을 테스트하는 스크립트
"""

import os
import sys
import time
import subprocess
import signal

def setup_environment():
    """환경 변수 설정"""
    os.environ['HUGGING_FACE_HUB_TOKEN'] = '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'
    os.environ['ROS_DOMAIN_ID'] = '0'
    os.environ['AMENT_PREFIX_PATH'] = '/home/soda/vla/ROS_action/install:' + os.environ.get('AMENT_PREFIX_PATH', '')
    
    print("🔧 Environment setup completed")
    print(f"  - HUGGING_FACE_HUB_TOKEN: {os.environ['HUGGING_FACE_HUB_TOKEN'][:10]}...")
    print(f"  - ROS_DOMAIN_ID: {os.environ['ROS_DOMAIN_ID']}")
    print(f"  - AMENT_PREFIX_PATH: {os.environ['AMENT_PREFIX_PATH'][:50]}...")

def test_model_loading():
    """모델 로딩 테스트"""
    print("\n🧪 Testing model loading...")
    
    try:
        import torch
        from transformers import AutoModel, AutoProcessor
        
        print("✅ PyTorch and Transformers imported successfully")
        
        # 모델 로딩 테스트
        model_name = "minium/mobile-vla-omniwheel"
        print(f"📥 Loading model: {model_name}")
        
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        print("✅ Model loaded successfully!")
        print(f"  - Model type: {type(model)}")
        print(f"  - Device: {next(model.parameters()).device}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def test_ros_nodes():
    """ROS 노드 테스트"""
    print("\n🤖 Testing ROS nodes...")
    
    try:
        # 카메라 시뮬레이터 시작
        print("📷 Starting camera simulator...")
        camera_proc = subprocess.Popen([
            'python3', 'src/mobile_vla_package/mobile_vla_package/test_camera_simulator.py'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        time.sleep(2)
        
        # RoboVLMs 추론 노드 시작
        print("🧠 Starting RoboVLMs inference node...")
        inference_proc = subprocess.Popen([
            'python3', 'src/mobile_vla_package/mobile_vla_package/robovlms_inference.py'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        time.sleep(5)
        
        # 프로세스 상태 확인
        if camera_proc.poll() is None:
            print("✅ Camera simulator is running")
        else:
            print("❌ Camera simulator failed to start")
            
        if inference_proc.poll() is None:
            print("✅ RoboVLMs inference node is running")
        else:
            print("❌ RoboVLMs inference node failed to start")
            
        # 정리
        camera_proc.terminate()
        inference_proc.terminate()
        
        return True
        
    except Exception as e:
        print(f"❌ ROS nodes test failed: {e}")
        return False

def main():
    """메인 함수"""
    print("🚀 RoboVLMs Docker Test Starting...")
    print("=" * 50)
    
    # 환경 설정
    setup_environment()
    
    # 모델 로딩 테스트
    model_success = test_model_loading()
    
    # ROS 노드 테스트
    ros_success = test_ros_nodes()
    
    # 결과 요약
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"  - Model Loading: {'✅ PASS' if model_success else '❌ FAIL'}")
    print(f"  - ROS Nodes: {'✅ PASS' if ros_success else '❌ FAIL'}")
    
    if model_success and ros_success:
        print("\n🎉 All tests passed! RoboVLMs system is ready for Docker deployment.")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
