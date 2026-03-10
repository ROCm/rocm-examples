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

# ============================================================================
# Python virtual environment (ready for ROCm wheel or tarball installation)
# ROCm installation is delegated to the CI workflow to support both methods
# ============================================================================

# Create virtual environment with base packages
RUN python3.13 -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install pyyaml cmake

# Set up virtual environment in PATH
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

# build ffmpeg from source
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
