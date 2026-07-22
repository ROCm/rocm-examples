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

FROM ubuntu:24.04

ARG ROCM_VERSION=7.14

ENV DEBIAN_FRONTEND=noninteractive

# ── System packages ───────────────────────────────────────────────────────────
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        sudo \
        gpg \
        libatomic1 \
        libquadmath0 \
        ca-certificates \
        git \
        wget \
        curl \
        build-essential \
        cmake \
        ninja-build \
        python3 \
        python3-venv \
        python3-pip \
        pkg-config \
        libglfw3-dev \
        libvulkan-dev \
        glslang-tools \
        vulkan-validationlayers \
        libopencv-dev \
        libavcodec-dev \
        libavformat-dev \
        libavutil-dev \
        libdw-dev \
    && rm -rf /var/lib/apt/lists/*

# Register ROCm repositories
RUN mkdir --parents --mode=0755 /etc/apt/keyrings && \
    wget https://repo.amd.com/rocm/packages-multi-arch/gpg/rocm.gpg -O - | \
    gpg --dearmor | tee /etc/apt/keyrings/amdrocm.gpg > /dev/null && \
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/amdrocm.gpg] https://repo.amd.com/rocm/packages-multi-arch/ubuntu2404 stable main" | tee /etc/apt/sources.list.d/rocm.list && \
    apt update

# Install ROCm
RUN apt install -y \
        amdrocm-core-sdk${ROCM_VERSION} \
        amdrocm-hiptensor-dev${ROCM_VERSION} \
        amdrocm-hiptensor-host${ROCM_VERSION} \
        amdrocm-rocalution-dev${ROCM_VERSION} \
        amdrocm-rocalution-host${ROCM_VERSION}

WORKDIR /workspace
CMD ["/bin/bash"]
