# syntax=docker/dockerfile:1
#
# Ubuntu 24.04 image for testing rocm-examples against the multi-arch ROCm
# pip install (TheRock nightlies).
#
# Build:
#   docker build -t rocm-examples-test .
#   # Install only your target arch (faster/smaller):
#   docker build --build-arg ROCM_EXTRAS="libraries,devel,device-gfx1100" \
#                -t rocm-examples-test .
#
# Run tests (see run-tests.sh):
#   ./run-tests.sh [/path/to/rocm-examples]

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
ENV VENV=/opt/rocm-venv
RUN python3 -m venv ${VENV}
ENV PATH="${VENV}/bin:${PATH}"

# ── ROCm pip install ──────────────────────────────────────────────────────────
# ROCM_INDEX_URL: override to pin a specific nightly date, e.g.
#   https://rocm.nightlies.amd.com/whl-multi-arch/
# ROCM_EXTRAS: comma-separated pip extras; use device-gfxNNNN to target one arch.
ARG ROCM_INDEX_URL=https://rocm.nightlies.amd.com/whl-multi-arch/
ARG ROCM_EXTRAS=libraries,devel,device-all

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir pyyaml && \
    pip install --no-cache-dir \
        --index-url "${ROCM_INDEX_URL}" \
        "rocm[${ROCM_EXTRAS}]" && \
    rocm-sdk init

# ── ROCm environment variables ────────────────────────────────────────────────
# Source: https://github.com/ROCm/TheRock/issues/1658
# ROCM_PATH points at _rocm_sdk_devel (headers + compiler + linker stubs).
ARG PYTHON_VERSION=3.12
ENV SITE=${VENV}/lib/python${PYTHON_VERSION}/site-packages
ENV ROCM_PATH=${SITE}/_rocm_sdk_devel

ENV HIP_PLATFORM=amd
ENV HIP_PATH=${ROCM_PATH}
ENV HIP_CLANG_PATH=${ROCM_PATH}/llvm/bin
ENV HIP_INCLUDE_PATH=${ROCM_PATH}/include
ENV HIP_LIB_PATH=${ROCM_PATH}/lib
ENV HIP_DEVICE_LIB_PATH=${ROCM_PATH}/lib/llvm/amdgcn/bitcode
ENV PATH="${ROCM_PATH}/bin:${ROCM_PATH}/llvm/bin:${PATH}"
ENV CPATH="${ROCM_PATH}/include"
ENV PKG_CONFIG_PATH="${ROCM_PATH}/lib/pkgconfig"
ENV LIBRARY_PATH="${ROCM_PATH}/lib:${ROCM_PATH}/lib64"

# Multi-arch layout note:
#   _rocm_sdk_devel/lib  — linker stubs (~41 KB each); GPU kernels externalized
#   _rocm_sdk_libraries/lib — full libs with kpack archives in .kpack/
#   _rocm_sdk_core/lib   — core runtime (HSA, OpenCL, etc.)
#
# _rocm_sdk_libraries/lib must appear before _rocm_sdk_devel/lib so the dynamic
# linker loads full libs (which can resolve kpack) instead of stubs.
ENV LD_LIBRARY_PATH="${SITE}/_rocm_sdk_core/lib:${SITE}/_rocm_sdk_libraries/lib:${SITE}/_rocm_sdk_devel/lib:${ROCM_PATH}/llvm/lib"

# ── Render group ──────────────────────────────────────────────────────────────
# GID must match the host render group so /dev/dri/renderD* is accessible.
# Override with: docker build --build-arg RENDER_GID=$(getent group render | cut -d: -f3)
ARG RENDER_GID=109
RUN groupadd --system --gid ${RENDER_GID} render 2>/dev/null || true && \
    useradd -m -G video,render developer && \
    mkdir -p /workspace && chown developer:developer /workspace

USER developer
WORKDIR /workspace
CMD ["/bin/bash"]
