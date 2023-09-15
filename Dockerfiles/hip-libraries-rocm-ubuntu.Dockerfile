# Ubuntu based docker image
FROM ubuntu:20.04

# Base packages that are required for the installation
RUN export DEBIAN_FRONTEND=noninteractive; \
    apt-get update -qq \
    && apt-get install --no-install-recommends -y \
        ca-certificates \
        git \
        locales-all \
        make \
        python3 \
        ssh \
        sudo \
        wget \
        pkg-config \
        glslang-tools \
        libvulkan-dev \
        vulkan-validationlayers \
        libglfw3-dev \
    && rm -rf /var/lib/apt/lists/*

ENV LANG en_US.utf8

# Install ROCM HIP and libraries using the installer script
RUN export DEBIAN_FRONTEND=noninteractive; \
    wget https://repo.radeon.com/amdgpu-install/5.6.1/ubuntu/focal/amdgpu-install_5.6.50601-1_all.deb \
    && apt-get update -qq \
    && apt-get install -y ./amdgpu-install_5.6.50601-1_all.deb \
    && rm ./amdgpu-install_5.6.50601-1_all.deb \
    && amdgpu-install -y --usecase=hiplibsdk --no-dkms \
    && apt-get install -y libnuma-dev \
    && rm -rf /var/lib/apt/lists/*

# Install CMake
RUN wget https://github.com/Kitware/CMake/releases/download/v3.21.7/cmake-3.21.7-linux-x86_64.sh \
    && mkdir /cmake \
    && sh cmake-3.21.7-linux-x86_64.sh --skip-license --prefix=/cmake \
    && rm cmake-3.21.7-linux-x86_64.sh

ENV PATH="/cmake/bin:/opt/rocm/bin:${PATH}"

RUN echo "/opt/rocm/lib" >> /etc/ld.so.conf.d/rocm.conf \
    && ldconfig

# Use render group as an argument from user
ARG GID=109

# Add the render group and a user with sudo permissions for the container
RUN groupadd --system --gid ${GID} render \
    && useradd -Um -G sudo,video,render developer \
    && echo developer ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/developer \
    && chmod 0440 /etc/sudoers.d/developer

RUN mkdir /workspaces && chown developer:developer /workspaces
WORKDIR /workspaces
VOLUME /workspaces

USER developer
