# syntax=docker/dockerfile:latest
# Above is required for substitutions in environment variables

# Ubuntu based docker image
FROM ubuntu:22.04

# The ROCm versions that this image is based of.
# Always write this down as major.minor.patch
ENV ROCM_VERSION=6.3.0
ENV ROCM_VERSION_APT=${ROCM_VERSION%.0}

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
        gnupg \
        g++ \
    && rm -rf /var/lib/apt/lists/*

ENV LANG en_US.utf8

# Install the HIP compiler and libraries from the ROCm repositories
RUN export DEBIAN_FRONTEND=noninteractive; \
    mkdir -p /etc/apt/keyrings \
    && wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor > /etc/apt/keyrings/rocm.gpg \
    && echo "deb [arch=amd64, signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/$ROCM_VERSION_APT/ jammy main" > /etc/apt/sources.list.d/rocm.list \
    && printf 'Package: *\nPin: origin "repo.radeon.com"\nPin-Priority: 9001\n' > /etc/apt/preferences.d/radeon.pref \
    && apt-get update -qq \
    && apt-get install --no-install-recommends -y \
        hip-base hipify-clang rocm-core hipcc \
        hip-dev rocm-hip-runtime-dev rocm-llvm-dev \
        rocrand-dev hiprand-dev \
        rocprim-dev hipcub-dev \
        rocblas-dev hipblas-dev \
        rocsolver-dev hipsolver-dev \
        rocfft-dev hipfft-dev \
        rocsparse-dev \
        rocthrust-dev \
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
