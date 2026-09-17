# hipThreads LLaMA 3 Inference Step 3: GPU Barrier Example

## Description

This example ports the step 2 barrier-synchronized CPU implementation to the AMD GPU using hipThreads. The transformer's `RunState` (activations and KV-cache) is moved to GPU-resident memory managed by rocThrust and hipC++ smart pointers. `std::thread` is replaced with `hip::thread`, `std::barrier` is replaced with `hip::barrier`, and `std::unique_ptr` on the GPU is replaced with `hip::std::unique_ptr`.

The source file `llama3.cxx` uses a `.cxx` extension but is compiled as HIP source via `set_source_files_properties`.

This example requires a downloaded LLaMA 3 model. See the model setup instructions in `../export.py`.

### Application flow

1. Parse arguments and memory-map model weights on the host.
2. Upload model weights to GPU device memory via `thrust::copy_n`.
3. Allocate GPU-resident activation buffers and KV-cache via `hip::std::unique_ptr`.
4. Load tokenizer on the host and encode the input prompt.
5. For each output token: run the GPU transformer forward pass using barrier-synchronized `hip::thread`s, then copy logits to the host to sample the next token.
6. Decode and print each token to stdout.

### Command line interface

Same as step 1. The application requires a model file and a tokenizer file.

- `-i <string>` Input prompt.
- `-m <string>` Run mode: `generate` (default) or `chat`.
- `-n <int>` Number of tokens to generate.
- `-t <float>` Sampling temperature.
- `-p <float>` Top-p sampling probability.
- `-s <int>` Random seed.
- `-z <string>` Path to tokenizer binary.
- `-y <string>` System prompt for chat mode.

## Key APIs and Concepts

### hipThreads

- `hip::thread` — GPU thread. Runs a `__device__` lambda on the GPU, analogous to `std::thread`.
- `hip::barrier` — synchronizes GPU threads between transformer sub-layers, analogous to `std::barrier`. Required because transformer layers must be computed sequentially even though computation within each layer is parallel.
- `hip::std::unique_ptr` — GPU-native smart pointer for device-resident allocations (from libhipcxx).

### rocThrust

- `thrust::device_malloc` — allocates raw device memory for model weights.
- `thrust::copy_n` — uploads model weights from the host memory-mapped file to device memory.

## Demonstrated API Calls

### hipThreads

- `hip::thread`
- `hip::barrier`
- `hip::std::unique_ptr`

### rocThrust

- `thrust::device_malloc`
- `thrust::copy_n`
