# Dockerfiles for the examples

This folder hosts Dockerfiles with ready-to-use environments for the various samples.
Each sample describes which environment it can be used with.

## Building

From this folder execute

```shell
docker build . -f <dockerfile> -t <result image name>
```

## List of Dockerfiles

### HIP libraries on the ROCm platform based on Ubuntu

Dockerfile: [hip-libraries-rocm-ubuntu.Dockerfile](hip-libraries-rocm-ubuntu.Dockerfile)

This is environment is based on Ubuntu targeting the ROCm platform. It has the
HIP runtime and the ROCm libraries installed. CMake is also installed in the image.
It can be used with most of the samples when running on a ROCm target.

### HIP libraries on the CUDA platform based on Ubuntu

Dockerfile: [hip-libraries-cuda-ubuntu.Dockerfile](hip-libraries-cuda-ubuntu.Dockerfile)

This is environment is based on Ubuntu targeting the CUDA platform. It has the
HIP runtime and the ROCm libraries installed. CMake is also installed in the image.
It can be used with the samples that support the CUDA target.
