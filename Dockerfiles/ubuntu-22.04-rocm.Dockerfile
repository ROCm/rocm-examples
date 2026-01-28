FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV GPU_TARGET=gfx1100
ENV THEROCK_FAMILY=gfx110X-all

# Install system dependencies
RUN apt-get update -qq && \
    apt-get install -y \
        git \
        wget \
        curl \
        xz-utils \
        build-essential \
        python3.11 \
        python3.11-venv \
        libdw-dev \
        libglfw3-dev \
        libvulkan-dev \
        glslang-tools \
        libtiff-dev \
        libopencv-dev && \
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
