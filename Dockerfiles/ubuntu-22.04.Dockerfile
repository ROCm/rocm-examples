FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# GPU_TARGET and THEROCK_FAMILY are set at workflow runtime, not in the base image

# Install system dependencies
RUN apt-get update -qq && \
    apt-get install -y \
        git \
        wget \
        curl \
        xz-utils \
        pkgconf \
        build-essential \
        software-properties-common \
        python3.11 \
        python3.11-venv \
        libdw-dev \
        libglfw3-dev \
        libvulkan-dev \
        glslang-tools \
        libtiff-dev \
        libopencv-dev \
        libavcodec-dev \
        libavformat-dev \
        libavutil-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# GCC 11.4 (Ubuntu 22.04 default) eagerly evaluates static_assert(false) in
# uninstantiated templates, which breaks hipDNN SDK headers. GCC 13 defers
# evaluation until instantiation (P2593R1).
RUN add-apt-repository -y ppa:ubuntu-toolchain-r/test && \
    apt-get update -qq && \
    apt-get install -y gcc-13 g++-13 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 130 \
        --slave /usr/bin/g++ g++ /usr/bin/g++-13 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# ============================================================================
# Python virtual environment (ready for ROCm wheel or tarball installation)
# ROCm installation is delegated to the CI workflow to support both methods
# ============================================================================

ENV VIRTUAL_ENV=/opt/venv
RUN python3.11 -m venv ${VIRTUAL_ENV}
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir pyyaml cmake

ENV HIP_PLATFORM=amd

WORKDIR /workspace

CMD ["/bin/bash"]
