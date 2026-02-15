  #docker build -t ros-whisper .
  #docker run --rm --net=host -it ros-whisper
  # Dockerfile

  ################################################################################
  # 1) 베이스 이미지: CUDA 지원 + Ubuntu 22.04
  ################################################################################
  FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

  ################################################################################
  # 2) 환경 변수 및 비대화형 모드 설정
  ################################################################################
  ENV DEBIAN_FRONTEND=noninteractive \
      LANG=ko_KR.UTF-8 \
      ROS_DISTRO=humble \
      WORKSPACE=/workspace 
  ################################################################################
  # 3) 기본 패키지 설치 (비-ROS 의존성 및 APT transport)
  ################################################################################
  RUN apt-get update && apt-get install -y --no-install-recommends \
        apt-transport-https \
        build-essential \
        curl gnupg2 lsb-release locales \
        ffmpeg libsndfile1 libportaudio2 portaudio19-dev \
        git python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && locale-gen ko_KR.UTF-8

  ################################################################################
  # 4) ROS 2 APT 저장소 등록 및 ROS 설치
  ################################################################################
  RUN set -eux; \
      # ROS 2 키 추가
      curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | apt-key add -; \
      # ROS 2 저장소 등록
      echo "deb [arch=$(dpkg --print-architecture)] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
        > /etc/apt/sources.list.d/ros2.list; \
      apt-get update; \
      apt-get install -y --no-install-recommends \
        ros-humble-ros-base \
        ros-humble-rmw-cyclonedds-cpp \
    && rm -rf /var/lib/apt/lists/*

  ################################################################################
  # 5) Python 라이브러리 및 빌드 도구 설치
  ################################################################################
  RUN python3 -m pip install --no-cache-dir --upgrade pip; \
      pip3 install --no-cache-dir \
        colcon-common-extensions \
        torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117 \
        transformers accelerate \
        sounddevice scipy librosa

  ################################################################################
  # 6) 워크스페이스 복사 및 빌드
  ################################################################################
  WORKDIR ${WORKSPACE}
  COPY ros2_ws ./ros2_ws
  RUN bash -lc "source /opt/ros/${ROS_DISTRO}/setup.bash && \
        cd ros2_ws && colcon build --symlink-install"

  ################################################################################
  # 7) 앱 파일 복사 및 bashrc 설정
  ################################################################################
  COPY whisper_test.py ./whisper_test.py
  COPY sample.wav      ./sample.wav
  RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /root/.bashrc \
  && echo "source ${WORKSPACE}/ros2_ws/install/setup.bash" >> /root/.bashrc \
  && echo "alias wt='python3 ${WORKSPACE}/whisper_test.py'" >> /root/.bashrc \
  && echo "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp" >> /root/.bashrc \
  && echo "export ROS_DOMAIN_ID=20" >> /root/.bashrc

  ################################################################################
  # 8) 기본 셸 실행
  ################################################################################
  CMD ["bash"]
