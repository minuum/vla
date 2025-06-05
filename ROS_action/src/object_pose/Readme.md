수정필요
---
camera_pub에서 sensor_msgs/Image 타입으로 발행하는 /camera/image_raw 토픽에서 이미지를 받아와서
yolov5s를 통해 탐지된 객체의 boundingbox을 x,y축으로 /2해서 중심을 구한 후
카메라 시야각의 양쪽은 -1에서 1로 정규화한 후에 각도를 계산?
거리는 객체 크기에 따라 계산
ros_action_msgs/ObjectInfo 타입 ObjectInfo 토픽으로 발행 
