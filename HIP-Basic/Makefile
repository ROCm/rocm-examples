# MIT License
#
# Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

EXAMPLES := \
  bandwidth \
  bit_extract \
  cooperative_groups \
  device_query \
  device_globals \
  dynamic_shared \
  events \
  gpu_arch \
  hello_world \
  hipify \
  inline_assembly \
  matrix_multiplication \
  moving_average \
  multi_gpu_data_transfer \
  occupancy \
  opengl_interop \
  runtime_compilation \
  saxpy \
  shared_memory \
  streams \
  static_host_library \
  texture_management \
  vulkan_interop \
  warp_shuffle

# Only supported on HIP (not CUDA).
ifneq ($(GPU_RUNTIME), CUDA)
  EXAMPLES += \
    assembly_to_executable \
    llvm_ir_to_executable \
    module_api \
    static_device_library
endif

all: $(EXAMPLES)

clean: TARGET=clean
clean: all

$(EXAMPLES):
	$(MAKE) -C $@ $(TARGET)

.PHONY: all clean $(EXAMPLES)
