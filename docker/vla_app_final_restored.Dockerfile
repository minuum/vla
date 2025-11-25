# VLA App Final - 복원된 Dockerfile
# 기존 vla_app_final:latest 이미지를 역공학하여 재구성

# 베이스 이미지: Jetson PyTorch (43.7GB 이미지 기반)
FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

# 메타데이터
LABEL maintainer="VLA Team"
LABEL description="Event-Triggered VLA System for K-Project"
LABEL version="1.0-restored"

# 사용자 설정
USER root

# 환경 변수 설정 (Docker history에서 추출)
ENV DEBIAN_FRONTEND=noninteractive
ENV LANGUAGE=en_US:en
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8
ENV CUDA_HOME=/usr/local/cuda
ENV NVCC_PATH=/usr/local/cuda/bin/nvcc
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV CUDAARCHS=87
ENV CUDA_ARCHITECTURES=87
ENV PATH="/root/.local/bin:/root/.cargo/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ENV PYTHONUNBUFFERED=1
ENV RMW_IMPLEMENTATION=rmw_cyclonedx_cpp
ENV HF_HOME=/root/.cache/huggingface

# Jetson 전용 환경 설정 (Docker history에서 확인된 내용)
ENV TAR_INDEX_URL=http://jetson.webredirect.org:8000/jp6/cu122
ENV PIP_INDEX_URL=http://jetson.webredirect.org/jp6/cu122
ENV PIP_TRUSTED_HOST=jetson.webredirect.org

# 1. 시스템 패키지 업데이트 및 기본 도구 설치
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libhdf5-serial-dev hdf5-tools libhdf5-dev \
        curl gnupg lsb-release \
    && rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# 2. ROS 2 Humble 설치
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        # X11/Qt 라이브러리
        libxcb-randr0-dev libxcb-xtest0-dev libxcb-xinerama0-dev \
        libxcb-shape0-dev libxcb-xkb-dev libqt5x11extras5 \
        libxkbcommon-x11-0 libgl1-mesa-glx \
        python3-pip \
        # ROS 2 패키지
        ros-humble-desktop \
        ros-dev-tools \
        python3-colcon-common-extensions \
        python3-rosdep \
        ros-humble-cv-bridge \
        python3-opencv \
        ros-humble-rmw-cyclonedx-cpp \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 3. Python 패키지 설치 (Docker history에서 확인된 버전)
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir \
        "transformers==4.52.3" \
        "tokenizers==0.21.1" \
        triton \
        Pillow \
    && rm -rf /root/.cache/pip

# 4. h5py 설치 (Docker history에서 확인됨)
RUN H5PY_SETUP_REQUIRES=0 pip3 install --no-cache-dir --verbose 'h5py<3.11'

# 5. OpenVLA 설치 (Docker history에서 확인됨)
RUN git clone --depth=1 --recursive https://github.com/dusty-nv/openvla /opt/openvla && \
    cd /opt/openvla && \
    pip3 install --verbose -e .

# 6. Accelerate 패치 (Docker history에서 확인됨)
RUN PYTHON_ROOT=`pip3 show accelerate | grep Location: | cut -d' ' -f2` && \
    ACCELERATE_STATE="$PYTHON_ROOT/accelerate/state.py" && \
    echo "patching $ACCELERATE_STATE to use distributed backend 'gloo' instead of 'nccl'" && \
    sed -i 's|self.backend = backend|self.backend = "gloo"|g' ${ACCELERATE_STATE} && \
    grep 'self.backend' $ACCELERATE_STATE

# 7. PyTorch CUDA 테스트 스크립트 추가
COPY pytorch_cuda_test.py /usr/local/bin/pytorch_cuda_test.py
RUN chmod +x /usr/local/bin/pytorch_cuda_test.py

# 8. .bashrc 설정 (Docker history에서 확인된 내용)
RUN echo "" >> /root/.bashrc && \
    echo "# ROS 2 Humble environment setup" >> /root/.bashrc && \
    echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc && \
    echo "export ROS_DOMAIN_ID=20" >> /root/.bashrc && \
    echo "" >> /root/.bashrc && \
    echo "# RoboVLMs Poetry Project Helper (기존 내용 유지)" >> /root/.bashrc && \
    echo "alias activate_robovlms='cd /workspace/RoboVLMs && source \$(poetry env info --path)/bin/activate'" >> /root/.bashrc && \
    echo "echo \"Hint: Run 'activate_robovlms' to navigate to /workspace/RoboVLMs and activate Poetry environment.\"" >> /root/.bashrc && \
    echo "" >> /root/.bashrc && \
    echo "# PyTorch & CUDA Test Alias (기존 내용 유지)" >> /root/.bashrc && \
    echo "alias torch_cuda_test='python3 /usr/local/bin/pytorch_cuda_test.py'" >> /root/.bashrc && \
    echo "echo \"Hint: Run 'torch_cuda_test' to check PyTorch and CUDA setup (after activating Poetry env if PyTorch is managed by Poetry).\"" >> /root/.bashrc

# 9. 작업 디렉토리 설정
WORKDIR /workspace/ROS_action

# 10. 기본 명령어
CMD ["bash"]

# 빌드 명령어:
# docker build -t vla_app_final_restored:latest -f vla_app_final_restored.Dockerfile .