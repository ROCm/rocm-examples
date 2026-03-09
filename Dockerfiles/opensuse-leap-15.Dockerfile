FROM opensuse/leap:15

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
        python312 \
        libglfw-devel \
        vulkan-devel \
        glslang-devel \
        ffmpeg-4-libavcodec-devel \
        ffmpeg-4-libavformat-devel \
        ffmpeg-4-libavutil-devel \
        libva-utils \
        Mesa-libva \
        libdw-devel && \
    zypper clean -a

# ============================================================================
# Python virtual environment (ready for ROCm wheel or tarball installation)
# ROCm installation is delegated to the CI workflow to support both methods
# ============================================================================

# Create virtual environment with base packages
RUN python3.12 -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install pyyaml cmake

# Set up virtual environment in PATH
ENV PATH="/opt/venv/bin:${PATH}"
ENV VIRTUAL_ENV="/opt/venv"

ENV LIBVA_DRIVERS_PATH="/usr/lib64/dri"
ENV HIP_PLATFORM=amd

WORKDIR /workspace

CMD ["/bin/bash"]
