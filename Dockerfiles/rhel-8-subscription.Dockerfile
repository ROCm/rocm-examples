FROM registry.access.redhat.com/ubi8/ubi:latest

ARG VULKAN_SDK_VERSION=1.4.335.0
ARG GLFW_VERSION=3.4
ARG GLSLANG_VERSION=13.1.1

# GPU_TARGET and THEROCK_FAMILY are set at workflow runtime, not in the base image
ENV VULKAN_SDK_VERSION=${VULKAN_SDK_VERSION}

# NOTE: This Dockerfile is for local testing only. Do not push the built image
# to a public registry — the subscription entitlements are baked into the layer.
ARG RHSM_USER
ARG RHSM_PASS
RUN subscription-manager register \
        --username="${RHSM_USER}" \
        --password="${RHSM_PASS}" && \
    subscription-manager attach --auto && \
    subscription-manager repos \
        --enable codeready-builder-for-rhel-8-x86_64-rpms && \
    dnf install -y \
        https://dl.fedoraproject.org/pub/epel/epel-release-latest-8.noarch.rpm \
        https://mirrors.rpmfusion.org/free/el/rpmfusion-free-release-8.noarch.rpm && \
    dnf update -y && \
    dnf install -y \
        unzip \
        xz \
        gcc \
        gcc-c++ \
        make \
        wget \
        git \
        curl \
        python3.11 \
        python3.11-pip \
        elfutils-devel \
        mesa-libGL-devel \
        wayland-devel \
        libxkbcommon-devel \
        libXcursor-devel \
        libXi-devel \
        libXinerama-devel \
        libXrandr-devel \
        ffmpeg-devel && \
    dnf clean all && \
    ln -s /usr/include/ffmpeg/libavcodec /usr/include/libavcodec && \
    ln -s /usr/include/ffmpeg/libavformat /usr/include/libavformat && \
    ln -s /usr/include/ffmpeg/libavutil /usr/include/libavutil && \
    subscription-manager unregister

# ============================================================================
# Python virtual environment (ready for ROCm wheel or tarball installation)
# ROCm installation is delegated to the CI workflow to support both methods
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

# Install Vulkan SDK (headers, libraries, and layers — prebuilt CLI tools are
# superseded by the glslang build above via PATH ordering)
ENV VULKAN_SDK=/opt/vulkan-sdk/${VULKAN_SDK_VERSION}/x86_64
RUN mkdir -p /opt/vulkan-sdk && \
    wget https://sdk.lunarg.com/sdk/download/${VULKAN_SDK_VERSION}/linux/vulkansdk-linux-x86_64-${VULKAN_SDK_VERSION}.tar.xz && \
    tar -xvf vulkansdk-linux-x86_64-${VULKAN_SDK_VERSION}.tar.xz -C /opt/vulkan-sdk && \
    rm vulkansdk-linux-x86_64-${VULKAN_SDK_VERSION}.tar.xz

ENV PATH="/usr/local/bin:${VULKAN_SDK}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${VULKAN_SDK}/lib"
ENV VK_ADD_LAYER_PATH="${VULKAN_SDK}/share/vulkan/explicit_layer.d"
ENV PKG_CONFIG_PATH="${VULKAN_SDK}/share/pkgconfig:${VULKAN_SDK}/lib/pkgconfig"

ENV HIP_PLATFORM=amd

WORKDIR /workspace

CMD ["/bin/bash"]
