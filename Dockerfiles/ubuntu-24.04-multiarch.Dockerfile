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
ENV VENV=/opt/rocm-venv
RUN python3 -m venv ${VENV}
ENV PATH="${VENV}/bin:${PATH}"

# ── ROCm pip install ──────────────────────────────────────────────────────────
# ROCM_INDEX_URL: override to pin a specific nightly date.
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
# Computed at image build time from the actual venv layout, so this Dockerfile
# works with any Python 3.x version without hardcoding the minor version.
#
# Multi-arch layout note:
#   _rocm_sdk_devel/lib     — linker stubs (~41 KB each); GPU kernels externalized
#   _rocm_sdk_libraries/lib — full libs with kpack archives in .kpack/
#   _rocm_sdk_core/lib      — core runtime (HSA, OpenCL, etc.)
#
# _rocm_sdk_libraries/lib must appear before _rocm_sdk_devel/lib so the dynamic
# linker loads full libs (which can resolve kpack) instead of stubs.
RUN python3 - <<'PYEOF'
import sysconfig
venv = "/opt/rocm-venv"
site = sysconfig.get_path("purelib", vars={"base": venv, "platbase": venv})
rocm = site + "/_rocm_sdk_devel"
core = site + "/_rocm_sdk_core/lib"
libs = site + "/_rocm_sdk_libraries/lib"
lines = [
    "ROCM_PATH=" + rocm,
    "HIP_PLATFORM=amd",
    "HIP_PATH=" + rocm,
    "HIP_CLANG_PATH=" + rocm + "/llvm/bin",
    "HIP_INCLUDE_PATH=" + rocm + "/include",
    "HIP_LIB_PATH=" + rocm + "/lib",
    "HIP_DEVICE_LIB_PATH=" + rocm + "/lib/llvm/amdgcn/bitcode",
    "PATH=" + rocm + "/bin:" + rocm + "/llvm/bin:/opt/rocm-venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "CPATH=" + rocm + "/include",
    "PKG_CONFIG_PATH=" + rocm + "/lib/pkgconfig",
    "LIBRARY_PATH=" + rocm + "/lib:" + rocm + "/lib64",
    "LD_LIBRARY_PATH=" + core + ":" + libs + ":" + rocm + "/lib:" + rocm + "/llvm/lib",
]
open("/etc/rocm-sdk.env", "w").write("\n".join(lines) + "\n")
PYEOF

# ── Entrypoint: source ROCm env before every command ─────────────────────────
RUN printf '#!/bin/bash\nset -a\n. /etc/rocm-sdk.env\nset +a\nexec "$@"\n' > /entrypoint.sh \
    && chmod +x /entrypoint.sh

# ── Render group + non-root user ──────────────────────────────────────────────
# GID must match the host render group so /dev/dri/renderD* is accessible.
# Override with: docker build --build-arg RENDER_GID=$(getent group render | cut -d: -f3)
ARG RENDER_GID=109
RUN groupadd --system --gid ${RENDER_GID} render 2>/dev/null || true && \
    useradd -m -G video,render developer && \
    mkdir -p /workspace && chown developer:developer /workspace

USER developer
WORKDIR /workspace
ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]
