
# MIGraphX - Torch Examples

# Summary
The examples in this subdirectory showcase the functionality of the [MIGraphX](https://github.com/ROCmSoftwarePlatform/torch_migraphx/tree/master) library. The examples build on Linux for the ROCm (AMD GPU) backend.

## Introduction

Quantization is an optimization tool that is commonly used to accelerate inferencing for deep learning models. There are many flavors of quantization. In the examples here, we show how the ROCm stack can be used to accelerate inference using [INT8 quantization](https://arxiv.org/pdf/1712.05877.pdf) on pretrained models. 

The quantization workflow consists of two main steps:

- Generate quantization parameters 
    
- Convert relevant operations in the model's computational graph to use the quantized datatype

### Generating quantization parameters

There are three main methods for computing quantization parameters:

- Dynamic Quantization:
    - Model weights are pre-quantized , input/activation quantization parameters are computed dynamically at runtime
        
- Static Post Training Quantization (PTQ):
    - Quantization parameters are computed via calibration. Calibration involves calculating statistical attributes for relevant model nodes using provided sample input data
        
- Static Quantization Aware Training (QAT):
  - Quantization parameters are calibrated during the training process

**Note**: All three of these techniques are supported by PyTorch (at least in a prototype form), and so the examples leverage PyTorch's quantization APIs to perform this step.

### Converting and executing the quantized model
As of the latest PyTorch release, there is no support for executing quantized models on GPUs directly through the framework. To execute these quantized models, use AMD's graph optimizer, MIGraphX, which is built using the ROCm stack. The [torch_migraphx](https://github.com/ROCmSoftwarePlatform/torch_migraphx) library provides a friendly interface for optimizing PyTorch models using the MIGraphX graph optimizer. 

The examples show how to use this library to convert and execute PyTorch quantized models on GPUs using MIGraphX.

## Torch-MIGraphX

Torch-MIGraphX integrates AMD's graph inference engine with the PyTorch ecosystem. It provides a `mgx_module` object that may be invoked in the same manner as any other torch module, but utilizes the MIGraphX inference engine internally. 

This library currently supports two paths for lowering:

- FX Tracing: Uses tracing API provided by the `torch.fx` library.
  
- Dynamo Backend: Importing torch_migraphx automatically registers the "migraphx" backend that can be used with the `torch.compile` API.

### Installation instructions

Refer to the [Torch_MIGraphX](https://github.com/ROCmSoftwarePlatform/torch_migraphx/blob/master/README.md) page for Docker and source installation instructions. 

