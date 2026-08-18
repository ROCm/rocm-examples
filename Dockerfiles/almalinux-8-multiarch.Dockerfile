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

FROM almalinux:8

ARG VULKAN_SDK_VERSION=1.4.335.0
ARG GLFW_VERSION=3.4
ARG GLSLANG_VERSION=13.1.1

# GPU_TARGET and THEROCK_FAMILY are set at workflow runtime, not in the base image
ENV VULKAN_SDK_VERSION=${VULKAN_SDK_VERSION}

# libquadmath: amdflang links against libquadmath.so.0 at runtime (TheRock#3290)
RUN dnf install -y dnf-plugins-core && \
    dnf config-manager --set-enabled powertools && \
    dnf update -y && \
    dnf install -y \
        unzip \
        xz \
        gcc \
        gcc-c++ \
        gcc-toolset-13-gcc-c++ \
        gcc-toolset-13-libstdc++-devel \
        make \
        wget \
        git \
        curl \
        nasm \
        pkgconf-pkg-config \
        python3.11 \
        python3.11-pip \
        elfutils-devel \
        opencv-devel \
        mesa-libGL-devel \
        wayland-devel \
        libxkbcommon-devel \
        libXcursor-devel \
        libXi-devel \
        libXinerama-devel \
        libXrandr-devel \
        libatomic \
        libquadmath && \
    dnf clean all

# GCC 8's libstdc++fs has an ABI-incompatible std::filesystem::path layout vs
# the one expected by TheRock's libhsa-runtime64.so (built with a newer GCC).
# GCC 13 from gcc-toolset-13 provides a compatible implementation.
ENV PATH="/opt/rh/gcc-toolset-13/root/usr/bin:${PATH}" \
    LD_LIBRARY_PATH="/opt/rh/gcc-toolset-13/root/usr/lib/gcc/x86_64-redhat-linux/13"

# ============================================================================
# Python virtual environment (ready for multi-arch ROCm installation)
# ROCm installation is delegated to the CI workflow (whl-multi-arch or tarball-multi-arch)
# ============================================================================

RUN python3.11 -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install pyyaml cmake

ENV PATH="/opt/venv/bin:${PATH}"
ENV VIRTUAL_ENV="/opt/venv"

# Build GLFW from source (not available in RHEL 8 repos)
WORKDIR /tmp
RUN wget https://github.com/glfw/glfw/releases/download/${GLFW_VERSION}/glfw-${GLFW_VERSION}.zip && \
    unzip glfw-${GLFW_VERSION}.zip && \
    cmake -S glfw-${GLFW_VERSION} -B glfw-${GLFW_VERSION}/build \
        -DGLFW_BUILD_EXAMPLES=OFF \
        -DGLFW_BUILD_TESTS=OFF \
        -DGLFW_BUILD_DOCS=OFF && \
    cmake --build glfw-${GLFW_VERSION}/build --target install && \
    rm -rf /tmp/glfw-${GLFW_VERSION}*

# Build glslang from source (Vulkan SDK prebuilt binaries require glibc 2.34+,
# but RHEL 8 ships glibc 2.28)
RUN git clone --branch ${GLSLANG_VERSION} --depth 1 https://github.com/KhronosGroup/glslang.git && \
    cmake -S glslang -B glslang/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DENABLE_OPT=OFF \
        -DENABLE_CTEST=OFF && \
    cmake --build glslang/build -j$(nproc) --target glslang-standalone && \
    cp glslang/build/StandAlone/glslang /usr/local/bin/glslangValidator && \
    rm -rf /tmp/glslang

# Install Vulkan SDK (headers, libraries, and layers)
ENV VULKAN_SDK=/opt/vulkan-sdk/${VULKAN_SDK_VERSION}/x86_64
RUN mkdir -p /opt/vulkan-sdk && \
    wget https://sdk.lunarg.com/sdk/download/${VULKAN_SDK_VERSION}/linux/vulkansdk-linux-x86_64-${VULKAN_SDK_VERSION}.tar.xz && \
    tar -xvf vulkansdk-linux-x86_64-${VULKAN_SDK_VERSION}.tar.xz -C /opt/vulkan-sdk && \
    rm vulkansdk-linux-x86_64-${VULKAN_SDK_VERSION}.tar.xz && \
    cp /usr/local/bin/glslangValidator ${VULKAN_SDK}/bin/glslangValidator

ENV PATH="${VULKAN_SDK}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${VULKAN_SDK}/lib:${LD_LIBRARY_PATH}"
ENV VK_ADD_LAYER_PATH="${VULKAN_SDK}/share/vulkan/explicit_layer.d"
ENV PKG_CONFIG_PATH="${VULKAN_SDK}/share/pkgconfig:${VULKAN_SDK}/lib/pkgconfig:/usr/local/lib/pkgconfig:/usr/local/lib64/pkgconfig"

# Build FFmpeg from source (not available in RHEL 8 repos)
WORKDIR /tmp
RUN wget https://ffmpeg.org/releases/ffmpeg-4.4.6.tar.xz && \
    tar -xvf ffmpeg-4.4.6.tar.xz && \
    cd ffmpeg-4.4.6 && \
    ./configure --enable-pic --enable-shared && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    rm -rf /tmp/ffmpeg-4.4.6*

ENV HIP_PLATFORM=amd

WORKDIR /workspace

CMD ["/bin/bash"]
