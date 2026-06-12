# syntax=docker/dockerfile:1
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
#
# Ubuntu 24.04 image for testing rocm-examples against the multi-arch ROCm
# pip install (TheRock nightlies).
#
# Build manually:
#   docker build -t rocm-examples-test-ubuntu-24.04 -f Dockerfile.ubuntu-24.04 .
#
# Run tests (preferred):
#   ./run-tests.sh --distro ubuntu-24.04 [/path/to/rocm-examples] [ctest-filter]

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# ── System packages ───────────────────────────────────────────────────────────
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
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

# ── Python venv ───────────────────────────────────────────────────────────────
# ROCm is installed at CI runtime (see build-rocm-examples-reusable.yml)
# so the venv only needs pip tooling here.
ENV VENV=/opt/rocm-venv
RUN python3 -m venv ${VENV}
ENV PATH="${VENV}/bin:${PATH}"

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir pyyaml cmake

# ── Render group + non-root user ──────────────────────────────────────────────
# GID must match the host render group so /dev/dri/renderD* is accessible.
# Override with: docker build --build-arg RENDER_GID=$(getent group render | cut -d: -f3)
ARG RENDER_GID=109
RUN groupadd --system --gid ${RENDER_GID} render 2>/dev/null || true && \
    useradd -m -G video,render developer && \
    mkdir -p /workspace && chown developer:developer /workspace

USER developer
WORKDIR /workspace
CMD ["/bin/bash"]
