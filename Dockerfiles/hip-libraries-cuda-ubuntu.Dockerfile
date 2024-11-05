# syntax=docker/dockerfile:latest
# Above is required for substitutions in environment variables

# CUDA based docker image
FROM nvidia/cuda:12.6.0-devel-ubuntu22.04

# The ROCm versions that this image is based of.
# Always write this down as major.minor.patch
ENV ROCM_VERSION=6.3.0
ENV ROCM_VERSION_APT=${ROCM_VERSION%.0}

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
    && rm -rf /var/lib/apt/lists/*

# Install the HIP compiler and libraries from the ROCm repositories
RUN export DEBIAN_FRONTEND=noninteractive; \
    mkdir -p /etc/apt/keyrings \
    && wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor > /etc/apt/keyrings/rocm.gpg \
    && echo "deb [arch=amd64, signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/$ROCM_VERSION_APT/ jammy main" > /etc/apt/sources.list.d/rocm.list \
    && printf 'Package: *\nPin: origin "repo.radeon.com"\nPin-Priority: 9001\n' > /etc/apt/preferences.d/radeon.pref \
    && apt-get update -qq \
    && apt-get install -y hip-base hipify-clang rocm-core hipcc hip-dev rocm-llvm-dev \
    && rm -rf /var/lib/apt/lists/*

# Install CMake
RUN wget https://github.com/Kitware/CMake/releases/download/v3.21.7/cmake-3.21.7-linux-x86_64.sh \
    && mkdir /cmake \
    && sh cmake-3.21.7-linux-x86_64.sh --skip-license --prefix=/cmake \
    && rm cmake-3.21.7-linux-x86_64.sh

ENV PATH="/cmake/bin:/opt/rocm/bin:${PATH}"

RUN echo "/opt/rocm/lib" >> /etc/ld.so.conf.d/rocm.conf \
    && ldconfig

ENV HIP_COMPILER=nvcc HIP_PLATFORM=nvidia HIP_RUNTIME=cuda

# Install rocRAND
RUN wget https://github.com/ROCm/rocRAND/archive/refs/tags/rocm-${ROCM_VERSION}.tar.gz -O rocrand.tar.gz \
    && mkdir rocrand \
    && tar -xf ./rocrand.tar.gz --strip-components 1 -C rocrand \
    && rm ./rocrand.tar.gz \
    && cmake -S ./rocrand -B ./rocrand/build \
        -D CMAKE_MODULE_PATH=/opt/rocm/lib/cmake/hip \
        -D BUILD_HIPRAND=OFF \
        -D CMAKE_INSTALL_PREFIX=/opt/rocm \
        -D NVGPU_TARGETS="50" \
    && cmake --build ./rocrand/build --target install \
    && rm -rf ./rocrand

# Install hipCUB
RUN wget https://github.com/ROCm/hipCUB/archive/refs/tags/rocm-${ROCM_VERSION}.tar.gz -O hipcub.tar.gz \
    && mkdir hipcub \
    && tar -xf ./hipcub.tar.gz --strip-components 1 -C hipcub \
    && rm ./hipcub.tar.gz \
    && cmake -S ./hipcub -B ./hipcub/build \
        -D CMAKE_MODULE_PATH=/opt/rocm/lib/cmake/hip \
        -D CMAKE_INSTALL_PREFIX=/opt/rocm \
    && cmake --build ./hipcub/build --target install \
    && rm -rf ./hipcub

# Install hipblas-common
RUN wget https://github.com/ROCm/hipBLAS-common/archive/refs/tags/rocm-${ROCM_VERSION}.tar.gz -O hipblas-common.tar.gz \
    && mkdir hipblas-common \
    && tar -xf ./hipblas-common.tar.gz --strip-components 1 -C hipblas-common \
    && rm ./hipblas-common.tar.gz \
    && cmake -S ./hipblas-common -B ./hipblas-common/build \
    && cmake --build ./hipblas-common/build --target install

# Install hipBLAS
RUN wget https://github.com/ROCm/hipBLAS/archive/refs/tags/rocm-${ROCM_VERSION}.tar.gz -O hipblas.tar.gz \
    && mkdir hipblas \
    && tar -xf ./hipblas.tar.gz --strip-components 1 -C hipblas \
    && rm ./hipblas.tar.gz \
    && CXXFLAGS=-D__HIP_PLATFORM_NVIDIA__ cmake -S ./hipblas -B ./hipblas/build \
        -D CMAKE_MODULE_PATH=/opt/rocm/lib/cmake/hip \
        -D CMAKE_INSTALL_PREFIX=/opt/rocm \
        -D HIP_PLATFORM=nvidia \
    && cmake --build ./hipblas/build --target install \
    && rm -rf ./hipblas

# Install hipSOLVER
RUN wget https://github.com/ROCm/hipSOLVER/archive/refs/tags/rocm-${ROCM_VERSION}.tar.gz -O hipsolver.tar.gz \
    && mkdir hipsolver \
    && tar -xf ./hipsolver.tar.gz --strip-components 1 -C hipsolver \
    && rm ./hipsolver.tar.gz \
    && cmake -S ./hipsolver -B ./hipsolver/build \
        -D CMAKE_MODULE_PATH=/opt/rocm/lib/cmake/hip \
        -D CMAKE_INSTALL_PREFIX=/opt/rocm \
        -D USE_CUDA=ON \
    && cmake --build ./hipsolver/build --target install \
    && rm -rf ./hipsolver

# Install hipRAND
RUN wget https://github.com/ROCm/hipRAND/archive/refs/tags/rocm-${ROCM_VERSION}.tar.gz -O hiprand.tar.gz \
    && mkdir hiprand \
    && tar -xf ./hiprand.tar.gz --strip-components 1 -C hiprand \
    && rm ./hiprand.tar.gz \
    && cmake -S ./hiprand -B ./hiprand/build \
        -D CMAKE_MODULE_PATH=/opt/rocm/lib/cmake/hip \
        -D CMAKE_INSTALL_PREFIX=/opt/rocm \
        -D BUILD_WITH_LIB=CUDA \
        -D NVGPU_TARGETS="50" \
    && cmake --build ./hiprand/build --target install \
    && rm -rf ./hiprand

# Install hipFFT
RUN wget https://github.com/ROCm/hipFFT/archive/refs/tags/rocm-${ROCM_VERSION}.tar.gz -O hipfft.tar.gz \
    && mkdir hipfft \
    && tar -xf ./hipfft.tar.gz --strip-components 1 -C hipfft \
    && rm ./hipfft.tar.gz \
    && cmake -S ./hipfft -B ./hipfft/build \
        -D CMAKE_MODULE_PATH=/opt/rocm/lib/cmake/hip \
        -D CMAKE_INSTALL_PREFIX=/opt/rocm \
        -D BUILD_WITH_LIB=CUDA \
    && cmake --build ./hipfft/build --target install \
    && rm -rf ./hipfft

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
