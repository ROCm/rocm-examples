# Running quantized ResNet50 via MIGraphX

## Summary

This example walks through the dynamo Post Training Quantization (PTQ) workflow for running a quantized model using torch_migraphx.

## Prerequisites

- You must follow the installation instructions for the torch_migraphx library in [AI/MIGraphX/Quantization](https://github.com/ROCm/rocm-examples/tree/develop/AI/MIGraphX/Quantization/) before using this example.

## Steps for running a quantized model using torch_migraphx

1. Use torch.export and quantize_pt2e APIs to perform quantization.

    **Note**: The export API call is considered a prototype feature at the time this tutorial is written. Some call signatures may be modified in the future.

    ```python
    import torch
    from torchvision import models
    from torch._export import capture_pre_autograd_graph
    from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
    ```

    ```python
    import torch_migraphx
    from torch_migraphx.dynamo.quantization import MGXQuantizer

    model_fp32 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).eval()
    input_fp32 = torch.randn(2, 3, 28, 28)

    torch_fp32_out = model_fp32(input_fp32)
    ```

    ```python
    model_export = capture_pre_autograd_graph(model_fp32, (input_fp32, ))
    ```

    Use the pt2e API to prepare, calibrate, and convert the model. Torch-MIGraphX provides a custom Quantizer for performing quantization that is compatible with MIGraphX.

    ```python
    quantizer = MGXQuantizer()
    m = prepare_pt2e(model_export, quantizer)

    # psudo calibrate
    with torch.no_grad():
      for _ in range(10):
        m(torch.randn(2, 3, 28, 28))
    q_m = convert_pt2e(m)
    torch_qout = q_m(input_fp32)
    ```

2. Lower Quantized model to MIGraphX. This step is the same as lowering any other model using torch.compile!

    ```python
    mgx_mod = torch.compile(q_m, backend='migraphx').cuda()
    mgx_out = mgx_mod(input_fp32.cuda())

    print(f"PyTorch FP32 (Gold Value):\n{torch_fp32_out}")
    print(f"PyTorch INT8 (Fake Quantized):\n{torch_qout}")
    print(f"MIGraphX INT8:\n{mgx_out}")
    ```

3. Performance

    Do a quick test to measure the performance gain from using quantization.

    ```python
    import copy
    import torch._dynamo

    # We will use this function to benchmark all modules:
    def benchmark_module(model, inputs, iterations=100):
        model(*inputs)
        torch.cuda.synchronize()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()

        for _ in range(iterations):
            model(*inputs)
        end_event.record()
        torch.cuda.synchronize()

        return start_event.elapsed_time(end_event) / iterations

    # Benchmark MIGraphX INT8
    mgx_int8_time = benchmark_module(mgx_mod, [input_fp32.cuda()])
    torch._dynamo.reset()

    # Benchmark MIGraphX FP32
    mgx_module_fp32 = torch.compile(copy.deepcopy(model_fp32), backend='migraphx').cuda()
    mgx_module_fp32(input_fp32.cuda())
    mgx_fp32_time = benchmark_module(mgx_module_fp32, [input_fp32.cuda()])
    torch._dynamo.reset()

    # Benchmark MIGraphX FP16
    mgx_module_fp16 = torch.compile(copy.deepcopy(model_fp32).half(), backend='migraphx').cuda()
    input_fp16 = input_fp32.cuda().half()
    mgx_module_fp16(input_fp16)
    mgx_fp16_time = benchmark_module(mgx_module_fp16, [input_fp16])

    print(f"{mgx_fp32_time=:0.4f}ms")
    print(f"{mgx_fp16_time=:0.4f}ms")
    print(f"{mgx_int8_time=:0.4f}ms")
    ```

 Note that these performance gains (or lack of gains) will vary depending on the specific hardware in use.
