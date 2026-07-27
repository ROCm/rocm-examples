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

FROM registry.suse.com/suse/sle15:15.7

ARG VULKAN_SDK_VERSION=1.4.335.0
ARG GLFW_VERSION=3.4

# GPU_TARGET and THEROCK_FAMILY are set at workflow runtime, not in the base image
ENV VULKAN_SDK_VERSION=${VULKAN_SDK_VERSION}

RUN zypper -qni update -y && \
    zypper -qni install -y \
        awk \
        unzip \
        xz \
        gcc \
        gcc-c++ \
        make \
        wget \
        git \
        curl \
        nasm \
        pkg-config \
        python313 \
        libdw-devel \
        Mesa-libGL-devel \
        wayland-devel \
        libxkbcommon-devel \
        libXcursor-devel \
        libXi-devel \
        libXinerama-devel \
        libXrandr-devel && \
    zypper clean -a

# GCC 7 (SLES 15 default) eagerly evaluates static_assert(false) in
# uninstantiated templates, which breaks hipDNN SDK headers. GCC 13 defers
# evaluation until instantiation (P2593R1).
# gcc13-c++ is not in the SLES base image and SUSEConnect requires registration,
# so we pull it from the openSUSE Leap 15.6 OSS repo (binary-compatible with SLE 15).
RUN zypper -qni addrepo -G https://download.opensuse.org/distribution/leap/15.6/repo/oss/ leap-oss && \
    zypper -qni install -y gcc13-c++ libstdc++6-devel-gcc13 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 130 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 130 && \
    update-alternatives --install /usr/bin/cc cc /usr/bin/gcc-13 130 && \
    update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++-13 130 && \
    zypper clean -a

# TheRock's tarball builds require elfutils >= 0.186 but SLES 15.7 ships 0.177.
WORKDIR /tmp
RUN zypper -qni install -y bzip2 m4 zlib-devel && \
    wget https://sourceware.org/elfutils/ftp/0.186/elfutils-0.186.tar.bz2 && \
    tar -xjf elfutils-0.186.tar.bz2 && \
    cd elfutils-0.186 && \
    ./configure --disable-debuginfod --disable-libdebuginfod && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    cd /tmp && \
    rm -rf elfutils-0.186* && \
    zypper clean -a

# ============================================================================
# Python virtual environment (ready for multi-arch ROCm installation)
# ROCm installation is delegated to the CI workflow (whl-multi-arch or tarball-multi-arch)
# ============================================================================

RUN python3.13 -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install pyyaml cmake

ENV PATH="/opt/venv/bin:${PATH}"
ENV VIRTUAL_ENV="/opt/venv"

# Build GLFW from source (not available in SLES repos)
WORKDIR /tmp
RUN wget https://github.com/glfw/glfw/releases/download/${GLFW_VERSION}/glfw-${GLFW_VERSION}.zip && \
    unzip glfw-${GLFW_VERSION}.zip && \
    cmake -S glfw-${GLFW_VERSION} -B glfw-${GLFW_VERSION}/build \
        -DGLFW_BUILD_EXAMPLES=OFF \
        -DGLFW_BUILD_TESTS=OFF \
        -DGLFW_BUILD_DOCS=OFF && \
    cmake --build glfw-${GLFW_VERSION}/build --target install && \
    rm -rf /tmp/glfw-${GLFW_VERSION}*

# Install Vulkan SDK
ENV VULKAN_SDK=/opt/vulkan-sdk/${VULKAN_SDK_VERSION}/x86_64
RUN mkdir -p /opt/vulkan-sdk && \
    wget https://sdk.lunarg.com/sdk/download/${VULKAN_SDK_VERSION}/linux/vulkansdk-linux-x86_64-${VULKAN_SDK_VERSION}.tar.xz && \
    tar -xvf vulkansdk-linux-x86_64-${VULKAN_SDK_VERSION}.tar.xz -C /opt/vulkan-sdk && \
    rm vulkansdk-linux-x86_64-${VULKAN_SDK_VERSION}.tar.xz

ENV PATH="${VULKAN_SDK}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${VULKAN_SDK}/lib"
ENV VK_ADD_LAYER_PATH="${VULKAN_SDK}/share/vulkan/explicit_layer.d"
ENV PKG_CONFIG_PATH="${VULKAN_SDK}/share/pkgconfig:${VULKAN_SDK}/lib/pkgconfig"

# Build FFmpeg from source (not available in SLES repos)
WORKDIR /tmp
RUN wget https://ffmpeg.org/releases/ffmpeg-4.4.6.tar.xz && \
    tar -xvf ffmpeg-4.4.6.tar.xz && \
    cd ffmpeg-4.4.6 && \
    ./configure --enable-pic --enable-shared && \
    make -j$(nproc) && \
    make install && \
    rm -rf /tmp/ffmpeg-4.4.6*

ENV HIP_PLATFORM=amd

WORKDIR /workspace

CMD ["/bin/bash"]
