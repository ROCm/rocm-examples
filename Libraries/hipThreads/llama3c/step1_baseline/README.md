# hipThreads LLaMA 3 Inference Step 1: CPU Baseline Example

## Description

This example implements autoregressive LLM inference for LLaMA 3 models on the CPU. It is a C++ adaptation of [Andrej Karpathy's llama2.c](https://github.com/karpathy/llama2.c), updated for LLaMA 3 by [jameswdelancey](https://github.com/jameswdelancey/llama3.c). The implementation loads model weights from a binary checkpoint file (exported from HuggingFace), runs a transformer forward pass for each token, and supports both generate and chat modes.

This is the CPU baseline for the LLaMA 3 porting series. Steps 2 through 4 progressively move computation to the GPU using hipThreads.

This example requires a downloaded LLaMA 3 model. See the model setup instructions in `../export.py` and `../README.md`.

### Application flow

1. Parse command-line arguments for model path, tokenizer path, temperature, number of steps, and run mode.
2. Memory-map the model weights from the binary checkpoint file.
3. Allocate activation and KV-cache buffers on the host as `std::unique_ptr<float[]>`.
4. Load the BPE tokenizer from the tokenizer file.
5. Encode the input prompt to token IDs.
6. For each output token: run the transformer forward pass (RMS norm, rope embeddings, attention, FFN) using `std::thread` for parallel matrix operations, then sample the next token and decode it to text.
7. Print tokens to stdout as they are generated.

### Command line interface

The application requires a model file and a tokenizer file. See `../export.py` for model export instructions.

- `-i <string>` Input prompt (generate mode).
- `-m <string>` Run mode: `generate` (default) or `chat`.
- `-n <int>` Number of tokens to generate. Default: `4096`.
- `-t <float>` Sampling temperature in [0, inf]. Default: `1.0`.
- `-p <float>` Top-p sampling probability in [0, 1]. Default: `0.9`.
- `-s <int>` Random seed. Default: current time.
- `-z <string>` Path to the tokenizer binary file.
- `-y <string>` System prompt for chat mode.

Example:

```shell
./hipthreads_llama3c_step1_baseline model.bin -z tokenizer.bin -i "Once upon a time" -n 100
```
