# syntax=docker/dockerfile:latest
# Above is required for substitutions in environment variables

# Ubuntu based docker image
FROM ubuntu:22.04

# The ROCm versions that this image is based of.
# Always write this down as major.minor.patch
ENV ROCM_VERSION=7.2.0
ENV ROCM_VERSION_APT=${ROCM_VERSION%.0}
ENV AMDGPU_INSTALLER_VERSION=7.2.70200-1

# Base packages that are required for the installation
RUN export DEBIAN_FRONTEND=noninteractive; \
    apt-get update -qq \
    && apt-get install --no-install-recommends -y \
        ca-certificates \
        git \
        locales-all \
        make \
        python3 \
        python3-pip \
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
        protobuf-compiler \
        libprotoc-dev \
        libopencv-dev \
        libavcodec-dev \
        libavformat-dev \
        libavutil-dev \
    && rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.utf8

# Install the HIP compiler and libraries from the ROCm repositories
# Use amdgpu-install to set up both the amdgpu and ROCm repos
RUN export DEBIAN_FRONTEND=noninteractive; \
    wget https://repo.radeon.com/amdgpu-install/$ROCM_VERSION_APT/ubuntu/jammy/amdgpu-install_${AMDGPU_INSTALLER_VERSION}_all.deb \
    && apt-get update -qq \
    && apt-get -y install ./amdgpu-install_${AMDGPU_INSTALLER_VERSION}_all.deb \
    && rm -f ./amdgpu-install_${AMDGPU_INSTALLER_VERSION}_all.deb \
    && apt-get update -qq \
    && apt-get install -y \
        hip-base hipify-clang rocm-core hipcc \
        hip-dev rocm-hip-runtime-dev rocm-llvm-dev \
        rocm-dev half \
        rocrand-dev hiprand-dev \
        rocprim-dev hipcub-dev \
        rocblas-dev hipblas-dev \
        rocsolver-dev hipsolver-dev \
        rocfft-dev hipfft-dev \
        rocsparse-dev hipsparse-dev \
        rocthrust-dev \
        rocal-dev \
        rocalution-dev \
        rocdecode-dev \
        rocjpeg-dev \
        rpp-dev \
        libdw-dev rocprofiler-sdk \
        rccl-dev \
        hipblaslt-dev \
        hiptensor-dev \
        rocwmma-dev \
        migraphx-dev \
        hipsparselt-dev \
        mivisionx-dev \
        composablekernel-dev \
    && rm -rf /var/lib/apt/lists/*

# Install CMake via pip for a modern version
RUN python3 -m pip install --no-cache-dir cmake

ENV PATH="/opt/rocm/bin:${PATH}"

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
