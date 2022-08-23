# ROCm Examples

## Repository Contents
- [Common](/Common/) contains common utility functionality shared between the examples.
- [Dockerfiles](/Dockerfiles/) hosts Dockerfiles with ready-to-use environments for the various samples. See [Dockerfiles/README.md](Dockerfiles/README.md) for details.
- [Libraries](/Libraries/)
    - [rocRAND](/Libraries/rocRAND/)
        - [simple_distributions_cpp](/Libraries/rocRAND/simple_distributions_cpp/): A command-line app to compare random number generation on the CPU and on the GPU with rocRAND.

## Prerequisites
### Linux
- [CMake](https://cmake.org/download/) (at least version 3.21)
- A number of examples also support building via  GNU Make - available through the distribution's package manager
- [ROCm](https://docs.amd.com/bundle/ROCm-Installation-Guide-v5.1.3/page/Overview_of_ROCm_Installation_Methods.html) (at least version 5.x.x)
- For example-specific prerequisites, see the example subdirectories.

### Windows
...

## Building the example suite
### Linux
These instructions assume that the prerequisites for every example are installed on the system.

#### CMake
- `$ git clone https://github.com/amd/rocm-examples.git`
- `$ cd rocm-examples`
- `$ cmake -S . -B build` (on ROCm) or `$ cmake -S . -B build -D GPU_RUNTIME=CUDA` (on CUDA)
- `$ cmake --build build`

#### Make
Beware that only a subset of the examples support building via Make.
- `$ git clone https://github.com/amd/rocm-examples.git`
- `$ cd rocm-examples`
- `$ make` (on ROCm) or `$ make GPU_RUNTIME=CUDA` (on CUDA)

### Linux with Docker
Alternatively, instead of installing the prerequisites on the system, the [Dockerfiles](/Dockerfiles/) in this repository can be used to build images that provide all required prerequisites. Note, that the ROCm kernel GPU driver still needs to be installed on the host system.

The following instructions showcase building the Docker image and full example suite inside the container using CMake:
- `$ git clone https://github.com/amd/rocm-examples.git`
- `$ cd rocm-examples/Dockerfiles`
- `$ docker build . -t rocm-examples -f hip-libraries-rocm-ubuntu.Dockerfile` (on ROCm) or `$ docker build . -t rocm-examples -f hip-libraries-cuda-ubuntu.Dockerfile` (on CUDA)
- `$ docker run -it --device /dev/kfd --device /dev/dri rocm-examples bash` (on ROCm) or `$ docker run -it --gpus=all rocm-examples bash` (on CUDA)
- `# git clone https://github.com/amd/rocm-examples.git`
- `# cd rocm-examples`
- `# cmake -S . -B build` (on ROCm) or `$ cmake -S . -B build -D GPU_RUNTIME=CUDA` (on CUDA)
- `# cmake --build build`

The built executables can be found and run in the `build` directory:
- `# ./build/Libraries/rocRAND/simple_distributions_cpp/simple_distributions_cpp`

### Windows
...
