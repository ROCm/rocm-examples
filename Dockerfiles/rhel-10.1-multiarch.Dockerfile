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
# RHEL 10.1 (UBI) image for testing rocm-examples against the multi-arch ROCm
# pip install (TheRock nightlies).
#
# Note: This uses the Red Hat UBI (Universal Base Image). Some packages
# available on full RHEL (GLFW, Vulkan dev headers, OpenCV) may not be
# available in UBI repos; those example categories will be skipped by CMake.
#
# Build manually:
#   docker build -t rocm-examples-test-rhel-10.1 -f Dockerfile.rhel-10.1 .
#
# Run tests (preferred):
#   ./run-tests.sh --distro rhel-10.1 [/path/to/rocm-examples] [ctest-filter]

FROM registry.access.redhat.com/ubi10/ubi:10.1

# ── GPG key + dnf plugins ─────────────────────────────────────────────────────
RUN rpm --import /etc/pki/rpm-gpg/RPM-GPG-KEY-redhat-release && \
    dnf install -y dnf-plugins-core && \
    dnf clean all

# ── Enable EPEL and CRB ───────────────────────────────────────────────────────
# Note: elfutils-devel is not available in UBI 10 even with EPEL + UBI-CRB.
# The code_object_isa_decode example (which requires libdw) will be skipped.
RUN dnf install -y wget && \
    wget --tries 5 https://dl.fedoraproject.org/pub/epel/epel-release-latest-10.noarch.rpm && \
    rpm -ivh epel-release-latest-10.noarch.rpm && \
    crb enable && \
    dnf clean all

# ── System packages ───────────────────────────────────────────────────────────
RUN dnf install -y --setopt=install_weak_deps=false \
        ca-certificates \
        git \
        curl \
        gcc \
        gcc-c++ \
        make \
        cmake \
        ninja-build \
        python3 \
        python3-pip \
        pkgconf-pkg-config \
        libatomic \
    && dnf clean all

# ── Python venv ───────────────────────────────────────────────────────────────
ENV VENV=/opt/rocm-venv
RUN python3 -m venv ${VENV}
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
RUN printf '#!/bin/bash\nset -a\n. /etc/rocm-sdk.env\nset +a\nexec "$@"\n' > /entrypoint.sh \
    && chmod +x /entrypoint.sh

# ── Render group + non-root user ──────────────────────────────────────────────
ARG RENDER_GID=109
RUN groupadd --system --gid ${RENDER_GID} render 2>/dev/null || groupmod -g ${RENDER_GID} render && \
    useradd -m -G video,render developer && \
    mkdir -p /workspace && chown developer:developer /workspace

USER developer
WORKDIR /workspace
ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]
