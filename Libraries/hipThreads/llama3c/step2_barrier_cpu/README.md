# hipThreads LLaMA 3 Inference Step 2: CPU Barrier Example

## Description

This example refactors the step 1 CPU baseline to use explicit barrier synchronization between transformer layers. Rather than joining all threads after each layer, threads synchronize via a `std::barrier`, allowing them to remain alive across multiple layers and reducing thread creation overhead. This refactoring also prepares the threading structure for the GPU port in step 3, where `hip::barrier` replaces `std::barrier`.

This step requires C++20 for `std::barrier`.

This example requires a downloaded LLaMA 3 model. See the model setup instructions in `../export.py`.

### Application flow

1. Parse arguments, memory-map model weights, allocate host buffers.
2. Load tokenizer and encode the input prompt.
3. Spawn worker threads that synchronize with `std::barrier` at each transformer sub-layer boundary, rather than being re-created per layer.
4. For each output token: run the barrier-synchronized transformer forward pass, sample the next token, and decode to text.
5. Print tokens to stdout as they are generated.

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
