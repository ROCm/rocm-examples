# CUDA based docker image
FROM nvidia/cuda:12.0.0-devel-ubuntu20.04

# Base packages that are required for the installation
RUN export DEBIAN_FRONTEND=noninteractive; \
    apt-get update -qq \
    && apt-get install --no-install-recommends -y \
        ca-certificates \
        git \
        locales-all \
        make \
        python3 \
        ssh \
        sudo \
        wget \
        pkg-config \
        glslang-tools \
        libvulkan-dev \
        vulkan-validationlayers \
        libglfw3-dev \
        gfortran \
        # Nvidia driver version needed for hipSOLVER's CUDA backend.
        # See https://docs.nvidia.com/deploy/cuda-compatibility/index.html#default-to-minor-version.
        nvidia-driver-455 \
    && rm -rf /var/lib/apt/lists/*

# Install HIP using the installer script
RUN export DEBIAN_FRONTEND=noninteractive; \
    wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | apt-key add - \
    && echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/6.0/ ubuntu main' > /etc/apt/sources.list.d/rocm.list \
    && apt-get update -qq \
    && apt-get install -y hip-base hipify-clang rocm-core hipcc hip-dev

# Install CMake
RUN wget https://github.com/Kitware/CMake/releases/download/v3.21.7/cmake-3.21.7-linux-x86_64.sh \
    && mkdir /cmake \
    && sh cmake-3.21.7-linux-x86_64.sh --skip-license --prefix=/cmake \
    && rm cmake-3.21.7-linux-x86_64.sh

ENV PATH="/cmake/bin:/opt/rocm/bin:${PATH}"

ENV HIP_COMPILER=nvcc HIP_PLATFORM=nvidia HIP_RUNTIME=cuda

RUN echo "/opt/rocm/lib" >> /etc/ld.so.conf.d/rocm.conf \
    && ldconfig

# Install rocRAND
RUN wget https://github.com/ROCm/rocRAND/archive/refs/tags/rocm-6.0.0.tar.gz \
    && tar -xf ./rocm-6.0.0.tar.gz \
    && rm ./rocm-6.0.0.tar.gz \
    && cmake -S ./rocRAND-rocm-6.0.0 -B ./rocRAND-rocm-6.0.0/build \
        -D CMAKE_MODULE_PATH=/opt/rocm/lib/cmake/hip \
        -D BUILD_HIPRAND=OFF \
        -D CMAKE_INSTALL_PREFIX=/opt/rocm \
        -D NVGPU_TARGETS="50" \
    && cmake --build ./rocRAND-rocm-6.0.0/build --target install \
    && rm -rf ./rocRAND-rocm-6.0.0

# Install hipCUB
RUN wget https://github.com/ROCm/hipCUB/archive/refs/tags/rocm-6.0.0.tar.gz \
    && tar -xf ./rocm-6.0.0.tar.gz \
    && rm ./rocm-6.0.0.tar.gz \
    && cmake -S ./hipCUB-rocm-6.0.0 -B ./hipCUB-rocm-6.0.0/build \
        -D CMAKE_MODULE_PATH=/opt/rocm/lib/cmake/hip \
        -D CMAKE_INSTALL_PREFIX=/opt/rocm \
    && cmake --build ./hipCUB-rocm-6.0.0/build --target install \
    && rm -rf ./hipCUB-rocm-6.0.0

# Install hipBLAS
# hipBLAS cmake for rocm-6.0.0 is broken added CXXFLAGS=-D__HIP_PLATFORM_NVIDIA__ as fix
RUN wget https://github.com/ROCm/hipBLAS/archive/refs/tags/rocm-6.0.0.tar.gz \
    && tar -xf ./rocm-6.0.0.tar.gz \
    && rm ./rocm-6.0.0.tar.gz \
    && CXXFLAGS=-D__HIP_PLATFORM_NVIDIA__ cmake -S ./hipBLAS-rocm-6.0.0 -B ./hipBLAS-rocm-6.0.0/build \
        -D CMAKE_MODULE_PATH=/opt/rocm/lib/cmake/hip \
        -D CMAKE_INSTALL_PREFIX=/opt/rocm \
        -D USE_CUDA=ON \
    && cmake --build ./hipBLAS-rocm-6.0.0/build --target install \
    && rm -rf ./hipBLAS-rocm-6.0.0

# Install hipSOLVER
# hipSOLVER cmake for rocm-6.0.0 is broken added CXXFLAGS=-D__HIP_PLATFORM_NVIDIA__ as fix
RUN wget https://github.com/ROCm/hipSOLVER/archive/refs/tags/rocm-6.0.0.tar.gz \
    && tar -xf ./rocm-6.0.0.tar.gz \
    && rm ./rocm-6.0.0.tar.gz \
    && CXXFLAGS=-D__HIP_PLATFORM_NVIDIA__ cmake -S ./hipSOLVER-rocm-6.0.0 -B ./hipSOLVER-rocm-6.0.0/build \
        -D CMAKE_MODULE_PATH=/opt/rocm/lib/cmake/hip \
        -D CMAKE_INSTALL_PREFIX=/opt/rocm \
        -D USE_CUDA=ON \
    && cmake --build ./hipSOLVER-rocm-6.0.0/build --target install \
    && rm -rf ./hipSOLVER-rocm-6.0.0

# Install hipRAND
# Build from commit that removes deprecated macro use
RUN git clone https://github.com/ROCm/hipRAND.git hipRAND-rocm-6.0.0 \
    && cd hipRAND-rocm-6.0.0 \
    && git reset --hard  4925f0da96fad5b9f532ddc79f1f52fc279d329f \
    && cmake -S . -B ./build \
        -D CMAKE_MODULE_PATH=/opt/rocm/lib/cmake/hip \
        -D CMAKE_INSTALL_PREFIX=/opt/rocm \
        -D BUILD_WITH_LIB=CUDA \
        -D NVGPU_TARGETS="50" \
    && cmake --build ./build --target install \
    && cd .. \
    && rm -rf ./hipRAND-rocm-6.0.0

# Install hipFFT
RUN wget https://github.com/ROCm/hipFFT/archive/refs/tags/rocm-6.0.0.tar.gz \
    && tar -xf ./rocm-6.0.0.tar.gz \
    && rm ./rocm-6.0.0.tar.gz \
    && cmake -S ./hipFFT-rocm-6.0.0 -B ./hipFFT-rocm-6.0.0/build \
        -D CMAKE_MODULE_PATH=/opt/rocm/lib/cmake/hip \
        -D CMAKE_INSTALL_PREFIX=/opt/rocm \
        -D BUILD_WITH_LIB=CUDA \
    && cmake --build ./hipFFT-rocm-6.0.0/build --target install \
    && rm -rf ./hipFFT-rocm-6.0.0

# Use render group as an argument from user
ARG GID=109

# Add the render group or change id if already exists
RUN if [ $(getent group render) ]; then \
        groupmod --gid ${GID} render; \
    else \
        groupadd --system --gid ${GID} render; \
    fi

# Add a user with sudo permissions for the container
RUN useradd -Um -G sudo,video,render developer \
    && echo developer ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/developer \
    && chmod 0440 /etc/sudoers.d/developer

RUN mkdir /workspaces && chown developer:developer /workspaces
WORKDIR /workspaces
VOLUME /workspaces

USER developer
