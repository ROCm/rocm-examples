# hipThreads LLaMA 3 Inference Step 4: Wavefront SIMD Example

## Description

This example extends the step 3 GPU barrier implementation with wavefront-level SIMD vectorization. GPU threads are created with `hip::wthread::max_width()` to fill a full wavefront width, and inner loops use `hip::this_thread::get_fiber_id()` and `hip::this_thread::get_width()` to stride across elements in SIMD-parallel fashion. This maps the inner matrix-vector and element-wise operations directly onto the GPU's hardware SIMD lanes, maximizing arithmetic throughput.

The barrier synchronization pattern from step 3 is preserved between sub-layers.

This example requires a downloaded LLaMA 3 model. See the model setup instructions in `../export.py`.

### Application flow

1. Parse arguments, upload model weights to GPU device memory.
2. Allocate GPU-resident activation and KV-cache buffers.
3. Load tokenizer and encode the input prompt.
4. For each output token: run the SIMD-vectorized GPU transformer forward pass, copy logits to host, sample and decode the next token.
5. Print each token to stdout.

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

- `hip::wthread::max_width()` — requests a thread with a full wavefront width of fibers, filling all SIMD lanes.
- `hip::this_thread::get_fiber_id()` — returns the fiber's 0-based lane index within the wavefront.
- `hip::this_thread::get_width()` — returns the total number of fibers, used as the stride in vectorized loops.
- `hip::barrier` — synchronizes wavefront-width threads at transformer sub-layer boundaries.

## Demonstrated API Calls

### hipThreads

- `hip::wthread`
- `hip::wthread::max_width`
- `hip::wthread::join`
- `hip::this_thread::get_fiber_id`
- `hip::this_thread::get_width`
- `hip::barrier`

### rocThrust

- `thrust::device_malloc`
- `thrust::copy_n`
