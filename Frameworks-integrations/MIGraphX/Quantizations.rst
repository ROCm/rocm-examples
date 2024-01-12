{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Run Quantized ResNet50 via MIGraphX\n",
    "This notebook walks through the dynamo PTQ workflow for running a quantized model using torch_migraphx."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Use torch.export and quantize_pt2e APIs to perform quantization\n",
    "Note: The export API call is considered a prototype feature at the time this tutorial is written. Some call signatures may be modified in the future."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "from torchvision import models\n",
    "from torch._export import capture_pre_autograd_graph\n",
    "from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e\n",
    "\n",
    "import torch_migraphx\n",
    "from torch_migraphx.dynamo.quantization import MGXQuantizer\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "model_fp32 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).eval()\n",
    "input_fp32 = torch.randn(2, 3, 28, 28)\n",
    "\n",
    "torch_fp32_out = model_fp32(input_fp32)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The `capture_pre_autograd_graph` call will be changed to a torch.export.export call once it supports the pre autograd capture functionallity. Currently there is a known issue when using only kwargs as inputs. See: https://github.com/pytorch/pytorch/issues/113744"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "model_export = capture_pre_autograd_graph(model_fp32, (input_fp32, ))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Use the pt2e API to prepare, calibrate and convert the model\n",
    "Torch-MIGraphX provides a custom Quantizer for performing quantization that is compatible with MIGraphX. Additional configs will also work as long as the configs ensure symmetric quantization using the signed int8 datatype. Currently, only symmetric quantization is supported in MIGraphX.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "quantizer = MGXQuantizer()\n",
    "m = prepare_pt2e(model_export, quantizer)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# psudo calibrate\n",
    "with torch.no_grad():\n",
    "    for _ in range(10):\n",
    "        m(torch.randn(2, 3, 28, 28))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "q_m = convert_pt2e(m)\n",
    "torch_qout = q_m(input_fp32)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Lower Quantized Model to MIGraphX\n",
    "This step is the same as lowering any other model using torch.compile!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "mgx_mod = torch.compile(q_m, backend='migraphx').cuda()\n",
    "mgx_out = mgx_mod(input_fp32.cuda())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(f\"PyTorch FP32 (Gold Value):\\n{torch_fp32_out}\")\n",
    "print(f\"PyTorch INT8 (Fake Quantized):\\n{torch_qout}\")\n",
    "print(f\"MIGraphX INT8:\\n{mgx_out}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Performance\n",
    "Let's do a quick test to measure the performance gain from using quantization. Note that these performance gains (or lack of gains) will vary depending on the specific hardware in use."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import copy\n",
    "import torch._dynamo"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# We will use this function to benchmark all modules:\n",
    "def benchmark_module(model, inputs, iterations=100):\n",
    "    model(*inputs)\n",
    "    torch.cuda.synchronize()\n",
    "\n",
    "    start_event = torch.cuda.Event(enable_timing=True)\n",
    "    end_event = torch.cuda.Event(enable_timing=True)\n",
    "\n",
    "    start_event.record()\n",
    "    for _ in range(iterations):\n",
    "        model(*inputs)\n",
    "    end_event.record()\n",
    "    torch.cuda.synchronize()\n",
    "\n",
    "    return start_event.elapsed_time(end_event) / iterations"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Benchmark MIGraphX INT8\n",
    "mgx_int8_time = benchmark_module(mgx_mod, [input_fp32.cuda()])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "torch._dynamo.reset()\n",
    "# Benchmark MIGraphX FP32\n",
    "mgx_module_fp32 = torch.compile(copy.deepcopy(model_fp32), backend='migraphx').cuda()\n",
    "mgx_module_fp32(input_fp32.cuda())\n",
    "mgx_fp32_time = benchmark_module(mgx_module_fp32, [input_fp32.cuda()])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "torch._dynamo.reset()\n",
    "# Benchmark MIGraphX FP16\n",
    "mgx_module_fp16 = torch.compile(copy.deepcopy(model_fp32).half(), backend='migraphx').cuda()\n",
    "input_fp16 = input_fp32.cuda().half()\n",
    "mgx_module_fp16(input_fp16)\n",
    "mgx_fp16_time = benchmark_module(mgx_module_fp16, [input_fp16])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(f\"{mgx_fp32_time=:0.4f}ms\")\n",
    "print(f\"{mgx_fp16_time=:0.4f}ms\")\n",
    "print(f\"{mgx_int8_time=:0.4f}ms\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "py_3.8",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.16"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
