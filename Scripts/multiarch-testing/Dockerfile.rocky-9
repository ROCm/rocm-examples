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
# Rocky Linux 9 image for testing rocm-examples against the multi-arch ROCm
# pip install (TheRock nightlies).
#
# Build manually:
#   docker build -t rocm-examples-test-rocky-9 -f Dockerfile.rocky-9 .
#
# Run tests (preferred):
#   ./run-tests.sh --distro rocky-9 [/path/to/rocm-examples] [ctest-filter]

FROM rockylinux:9

# ── Enable CRB and EPEL ───────────────────────────────────────────────────────
RUN dnf install -y epel-release && \
    dnf config-manager --set-enabled crb && \
    dnf clean all

# ── System packages ───────────────────────────────────────────────────────────
RUN dnf install -y --setopt=install_weak_deps=false \
        ca-certificates \
        git \
        wget \
        gcc-toolset-13-gcc \
        gcc-toolset-13-gcc-c++ \
        make \
        cmake \
        ninja-build \
        python3.12 \
        pkgconf-pkg-config \
        libatomic \
        elfutils-devel \
        glfw-devel \
        vulkan-devel \
        glslang \
        vulkan-validation-layers \
        opencv-devel \
        libdwarf-devel \
    && dnf clean all

# ── Python venv ───────────────────────────────────────────────────────────────
ENV VENV=/opt/rocm-venv
RUN python3.12 -m venv ${VENV}
ENV PATH="${VENV}/bin:${PATH}"

# ── ROCm pip install ──────────────────────────────────────────────────────────
ARG ROCM_INDEX_URL=https://rocm.nightlies.amd.com/whl-multi-arch/
ARG ROCM_EXTRAS=libraries,devel,device-all

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir pyyaml && \
    pip install --no-cache-dir \
        --index-url "${ROCM_INDEX_URL}" \
        "rocm[${ROCM_EXTRAS}]" && \
    rocm-sdk init

# ── ROCm environment variables ────────────────────────────────────────────────
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
RUN printf '#!/bin/bash\n. /opt/rh/gcc-toolset-13/enable\nset -a\n. /etc/rocm-sdk.env\nset +a\nexec "$@"\n' > /entrypoint.sh \
    && chmod +x /entrypoint.sh

# ── Render group + non-root user ──────────────────────────────────────────────
ARG RENDER_GID=109
RUN groupadd --system --gid ${RENDER_GID} render 2>/dev/null || true && \
    useradd -m -G video,render developer && \
    mkdir -p /workspace && chown developer:developer /workspace

USER developer
WORKDIR /workspace
ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]
