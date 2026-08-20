# hipThreads LLaMA 3 Inference Examples

## Description

This series implements autoregressive LLM inference for LLaMA 3 models. It is a C++ adaptation of [Andrej Karpathy's llama2.c](https://github.com/karpathy/llama2.c), updated for LLaMA 3 by [jameswdelancey](https://github.com/jameswdelancey/llama3.c). The implementation loads model weights from a binary checkpoint file, runs a transformer forward pass for each token, and supports both generate and chat modes.

The series starts from a CPU baseline using `std::thread` and progressively ports the transformer to the AMD GPU using hipThreads, introducing barrier synchronization and wavefront SIMD along the way.

## Prerequisites

In addition to the [hipThreads prerequisites](../README.md#prerequisites), this series requires a LLaMA 3 model exported to the binary checkpoint format expected by the C++ examples.

### Model setup

- Install the Python dependencies:

  ```shell
  pip install -r requirements.txt
  ```

- Download a LLaMA 3 checkpoint from HuggingFace and export it to a binary file using `export.py`:

  ```shell
  python export.py model.bin --hf <path-to-hf-model>
  ```

- Export the tokenizer to a binary file using `tokenizer.py`:

  ```shell
  python tokenizer.py --tokenizer-model <path-to-tokenizer.model>
  ```

Pass `model.bin` as the positional argument and `-z tokenizer.bin` to each step's executable. See the individual step READMEs for the full command-line interface.

### Steps

| Step | Directory | Description |
|------|-----------|-------------|
| 1 | `step1_baseline/` | CPU baseline: `std::thread` parallelizes matrix operations within each transformer layer. |
| 2 | `step2_barrier_cpu/` | CPU barrier: threads persist across layers using `std::barrier`, avoiding repeated join and spawn. |
| 3 | `step3_barrier_gpu/` | GPU port: `std::thread` replaced with `hip::thread`, `std::barrier` with `hip::barrier`, and activations moved to GPU-resident memory. |
| 4 | `step4_simdize/` | Wavefront SIMD: `hip::wthread`s are widened to fill a full wavefront, with each fiber handling a strided portion of the inner loop. |
