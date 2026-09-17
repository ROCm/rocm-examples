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
  runtime_compilation \
  saxpy \
  shared_memory \
  streams \
  static_host_library \
  texture_management \
  warp_shuffle

# Skips are driven by .github/build_tools/skip_manifest.py: CI runs
# generate_skip_tests.py and passes SKIP_FROM_BUILD / SKIP_FROM_TEST as
# repo-relative paths on the make command line (overriding these defaults).
# skip_here matches only entries naming THIS directory's examples. A bare local
# build/test skips nothing. The `test` target honors SKIP_FROM_BUILD too: an
# example that isn't built can't be tested, so build skips imply test skips.
SKIP_FROM_BUILD ?=
SKIP_FROM_TEST ?=
skip_here = $(foreach e,$(EXAMPLES),$(if $(filter %/$(notdir $(CURDIR))/$(e) $(notdir $(CURDIR))/$(e),$(1)),$(e)))

ifneq ($(wildcard /usr/include/GLFW/glfw3.h),)
  EXAMPLES += opengl_interop
  ifneq ($(wildcard /usr/include/vulkan/vulkan.h),)
    EXAMPLES += vulkan_interop
  endif
endif

# Only supported on HIP (not CUDA).
ifneq ($(GPU_RUNTIME), CUDA)
  EXAMPLES += \
    assembly_to_executable \
    cooperative_groups_double_buffered_tile \
    cooperative_groups_prefix_sum \
    llvm_ir_to_executable \
    module_api \
    static_device_library
endif

all: $(filter-out $(call skip_here,$(SKIP_FROM_BUILD)),$(EXAMPLES))

clean: TARGET=clean
clean: all

$(EXAMPLES):
	$(MAKE) -C $@ $(TARGET)

test: TARGET=test
test: $(filter-out $(call skip_here,$(SKIP_FROM_TEST) $(SKIP_FROM_BUILD)),$(EXAMPLES))

.PHONY: all clean test $(EXAMPLES)
