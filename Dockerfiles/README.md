# Dockerfiles for the examples

This folder hosts Dockerfiles with ready-to-use environments for the various samples.
Each sample describes which environment it can be used with.

## Building

From this folder execute

```shell
docker build . -f <dockerfile> -t <result image name>
```

## List of Dockerfiles

### ROCm on Ubuntu 24.04

Dockerfile: [ubuntu-24.04-rocm.Dockerfile](ubuntu-24.04-rocm.Dockerfile)

This is environment is based on Ubuntu targeting the ROCm platform. It has the
HIP runtime and the ROCm libraries installed. CMake is also installed in the image.
It can be used with most of the samples when running on a ROCm target.

### HIP libraries on the CUDA platform based on Ubuntu

Dockerfile: [hip-libraries-cuda-ubuntu.Dockerfile](hip-libraries-cuda-ubuntu.Dockerfile)

This is environment is based on Ubuntu targeting the CUDA platform. It has the
HIP runtime and the ROCm libraries installed. CMake is also installed in the image.
It can be used with the samples that support the CUDA target.

### ROCm installed via apt based on Ubuntu

Dockerfile: [ubuntu-24.04-rocm.Dockerfile](ubuntu-24.04-rocm.Dockerfile)

This is an Ubuntu 24.04 image with ROCm installed from the AMD apt repositories, along with the
system and Python dependencies needed to build rocm-examples. It registers both
the stable (`repo.amd.com`) and prerelease (`rocm.prereleases.amd.com`) repos;
the prerelease repo is pinned to priority 1 so only packages absent from stable
are pulled from it. This is used to install test packages such as `amdrocm-decode-test`
(prerelease-only, `7.14.0~pre3`), which provides the video and utility assets the
rocDecode examples need to build and run their tests.

### CI base images (multi-arch)

Lightweight base images for CI workflows based on various supported Linux distributions. The images include a Python virtual environment and complete system dependencies to build rocm-examples from source, but they **do not include any ROCm installation** — ROCm is installed at CI runtime (multi-arch wheel or tarball) by the build workflow.

Dockerfiles:

- [ubuntu-24.04-multiarch.Dockerfile](ubuntu-24.04-multiarch.Dockerfile)
- [ubuntu-26.04-multiarch.Dockerfile](ubuntu-26.04-multiarch.Dockerfile)
- [almalinux-8-multiarch.Dockerfile](almalinux-8-multiarch.Dockerfile)
- [sles-15.7-multiarch.Dockerfile](sles-15.7-multiarch.Dockerfile)
