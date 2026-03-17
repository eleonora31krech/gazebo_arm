FROM osrf/ros:jazzy-desktop

ENV DEBIAN_FRONTEND=noninteractive
ENV QT_X11_NO_MITSHM=1
ENV LIBGL_ALWAYS_SOFTWARE=0
ENV __NV_PRIME_RENDER_OFFLOAD=1
ENV __GLX_VENDOR_LIBRARY_NAME=nvidia

RUN apt-get update && apt-get install -y \
    git \
    nano \
    vim \
    curl \
    wget \
    lsb-release \
    gnupg2 \
    software-properties-common \
    build-essential \
    sudo \
    python3-pip \
    python3-colcon-common-extensions \
    python3-numpy \
    python3-rosdep \
    x11-apps \
    mesa-utils \
    ffmpeg \
    libglu1-mesa \
    libxrandr2 \
    libxi6 \
    libxrender1 \
    libxkbcommon-x11-0 \
    libxcb-cursor0 \
    libx11-xcb1 \
    ros-jazzy-xacro \
    ros-jazzy-robot-state-publisher \
    ros-jazzy-joint-state-publisher \
    ros-jazzy-joint-state-publisher-gui \
    ros-jazzy-ros2-control \
    ros-jazzy-ros2-controllers \
    ros-jazzy-controller-manager \
    ros-jazzy-vision-msgs \
    ros-jazzy-moveit \
    ros-jazzy-moveit-py \
    ros-jazzy-moveit-ros-planning-interface \
    ros-jazzy-moveit-planners \
    ros-jazzy-moveit-simple-controller-manager \
    ros-jazzy-moveit-visual-tools \
    ros-jazzy-ros-gz \
    ros-jazzy-ros-gz-sim \
    ros-jazzy-ros-gz-bridge \
    ros-jazzy-ros-gz-image \
    ros-jazzy-ros-gz-interfaces \
    ros-jazzy-gz-ros2-control \
    && rm -rf /var/lib/apt/lists/*


RUN pip3 install --no-cache-dir --break-system-packages --upgrade setuptools

RUN pip3 install --no-cache-dir --break-system-packages \
    lxml \
    transforms3d \
    ikpy


RUN pip3 uninstall -y numpy || true && \
    pip3 uninstall -y numpy || true && \
    rm -rf /usr/local/lib/python3.12/dist-packages/numpy* && \
    apt-get update && \
    apt-get install --reinstall -y python3-numpy && \
    python3 -c "import numpy; print('System numpy version:', numpy.__version__)" && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages ultralytics --ignore-installed numpy


WORKDIR /workspace

RUN echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc


COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]