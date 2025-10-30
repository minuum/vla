#!/usr/bin/env python3
# 사용자 제공 CSI 카메라 테스트 코드 (시스템 OpenCV 4.5.4 버전)

import cv2

gst_str = (
    "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, "
    "format=NV12, framerate=30/1 ! nvvidconv ! video/x-raw, format=BGRx ! "
    "videoconvert ! video/x-raw, format=BGR ! appsink"
)

print("🔧 시스템 OpenCV 4.5.4로 CSI 카메라 테스트 시작")
print(f"🔧 GStreamer 파이프라인: {gst_str}")

cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("❌ 카메라 열기 실패")
    exit(1)
else:
    print("✅ 카메라 열기 성공")
    
    for i in range(5):
        print(f"📸 프레임 읽기 시도 {i+1}/5...")
        ret, frame = cap.read()
        if ret and frame is not None:
            cv2.imwrite(f"frame_{i}.jpg", frame)
            print(f"✅ frame_{i}.jpg 저장 완료")
            print(f"   프레임 크기: {frame.shape}")
        else:
            print(f"❌ 프레임 {i+1} 읽기 실패 (ret={ret})")
            
    cap.release()
    print("🎉 CSI 카메라 테스트 완료")