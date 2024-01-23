
# MIGraphX - Torch Examples

# Summary
The examples in this subdirectory showcase the functionality of the [MIGraphX](https://github.com/ROCmSoftwarePlatform/torch_migraphx/tree/master) library. The examples build on Linux for the ROCm (AMD GPU) backend.

## Introduction

Quantization is an optimization tool that is commonly used to accelerate inferencing for deep learning models. There are many flavors of quantization. In the examples here, we show how the ROCm stack can be used to accelerate inference using [INT8 quantization](https://arxiv.org/pdf/1712.05877.pdf) on pretrained models. 

The quantization workflow consists of two main steps:

- Generate quantization parameters 
    
- Convert relevant operations in the model's computational graph to use the quantized datatype

### Generating Quantization Parameters

There are 3 main methods for computing quantization parameters:

- Dynamic Quantization:
    
        ○ Model weights are pre-quantized , input/activation quantization parameters are computed dynamically at runtime
        
- Static Post Training Quantization (PTQ):
    
        ○ Quantization parameters are computed via calibration. Calibration involves calculating statistical attributes for relevant model nodes using provided sample input data
        
- Static Quantization Aware Training (QAT):
    
        ○ Quantization parameters are calibrated during the training process

All 3 of these techniques are supported by PyTorch (at least in a prototype form), and so the examples leverage PyTorch's quantization APIs to perform this step.

### Converting and Executing the Quantized Model
As of the latest PyTorch release, there is no support for executing quantized models on GPUs directly through the framework. To execute these quantized models we use AMD's graph optimizer, MIGraphX, which is built using the rocm stack. The [torch_migraphx](https://github.com/ROCmSoftwarePlatform/torch_migraphx) library provides a friendly interface for optimizing PyTorch models using the MIGraphX graph optimizer. The examples show how to use this library to convert and execute PyTorch quantized models on GPUs using MIGraphX.

## Torch-MIGraphX

Torch-MIGraphX integrates AMD's graph inference engine with the PyTorch ecosystem. It provides a `mgx_module` object that may be invoked in the same manner as any other torch module, but utilizes the MIGraphX inference engine internally. 

This library currently supports two paths for lowering:
1. FX Tracing: Uses tracing API provided by the `torch.fx` library.
2. Dynamo Backend: Importing torch_migraphx automatically registers the "migraphx" backend that can be used with the `torch.compile` API.


## Getting started
### Docker
The simplest and recommended way to get started is using the provided Dockerfile.
Build using:
```
./build_image.sh
```
Start container using:
```
sudo docker run -it --network=host --device=/dev/kfd --device=/dev/dri --group-add=video --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined torch_migraphx
```

The default Dockerfile builds on the nightly pytorch container and installs the latest source version of MIGraphX and torch_migraphx. For more builds refer to the docker directory.

### Installing from source
Install Pre-reqs:
- [PyTorch (ROCM version)](https://rocm.docs.amd.com/projects/install-on-linux/en/develop/how-to/3rd-party/pytorch-install.html#using-a-wheels-package)
- [MIGraphX](https://github.com/ROCm/AMDMIGraphX?tab=readme-ov-file#installing-from-binaries)

Build and install from source
```
git clone https://github.com/ROCmSoftwarePlatform/torch_migraphx.git
cd ./torch_migraphx/py
export TORCH_CMAKE_PATH=$(python -c "import torch; print(torch.utils.cmake_prefix_path)")
pip install .
```

## Example usage
```
# FX Tracing
torch_migraphx.fx.lower_to_mgx(torch_model, sample_inputs)

# Dynamo Backend
torch.compile(torch_model, backend="migraphx")
```

### Lower resnet50 using FX Tracing
```
import torch
import torchvision
import torch_migraphx

resnet = torchvision.models.resnet50()
sample_input = torch.randn(2, 3, 64, 64)
resnet_mgx = torch_migraphx.fx.lower_to_mgx(resnet, [sample_input])
result = resnet_mgx(sample_input)
```

### Lower densenet using torch.compile
```
import torch
import torchvision
import torch_migraphx

densenet = torchvision.models.densenet161().cuda()
sample_input = torch.randn(2, 3, 512, 512).cuda()
densenet_mgx = torch.compile(densenet, backend="migraphx")
result = densenet_mgx(sample_input.cuda())
```



