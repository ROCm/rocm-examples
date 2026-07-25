# syntax=docker/dockerfile:latest
# Above is required for substitutions in environment variables

# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
        rocm \
        rocal-dev \
        rocdecode-dev \
        rocjpeg-dev \
        libdw-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages (cmake and deps for Libraries build)
RUN python3 -m pip install --no-cache-dir --upgrade pip \
    && python3 -m pip install --no-cache-dir cmake future==1.0.0 pytz==2022.1 numpy==1.23.0 \
        google==3.0.0 protobuf==3.12.4

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
