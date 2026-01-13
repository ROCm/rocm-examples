# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

EXAMPLE := ComposableKernel_ck_tile_permute
EXAMPLE_ALT := ComposableKernel_ck_tile_permute_alternative
GPU_RUNTIME := HIP
COMMON_INCLUDE_DIR := ../../../../Common

# Supported GPU architectures for ComposableKernel
SUPPORTED_HIP_ARCHITECTURES := gfx90a gfx942
include $(COMMON_INCLUDE_DIR)/filter_hip_architectures.mk

# HIP variables
ROCM_INSTALL_DIR := /opt/rocm
HIP_INCLUDE_DIR  := $(ROCM_INSTALL_DIR)/include

HIPCXX ?= $(ROCM_INSTALL_DIR)/bin/hipcc

# Common variables and flags
CXX_STD   := c++17
ICXXFLAGS := -std=$(CXX_STD) -x hip
ICPPFLAGS :=
ILDFLAGS  :=
ILDLIBS   :=

CXXFLAGS ?= -Wall -Wextra -Wno-undefined-func-template -Wno-float-equal
ICXXFLAGS += $(CXXFLAGS)
ICPPFLAGS += $(CPPFLAGS)
ILDFLAGS  += $(LDFLAGS)
ILDLIBS   += $(LDLIBS)

ALT_DEFS := -DPERMUTE_USE_ALTERNATIVE_IMPL

.PHONY: all
ifeq ($(SKIP_BUILD),1)
all:
	@echo "Skipping $(EXAMPLE): no supported architectures"
else
all: $(EXAMPLE) $(EXAMPLE_ALT)
endif

permute_base.o: permute.cpp
	$(HIPCXX) $(ICXXFLAGS) $(ICPPFLAGS) -o $@ -c $<

permute_alt.o: permute.cpp
	$(HIPCXX) $(ICXXFLAGS) $(ICPPFLAGS) $(ALT_DEFS) -o $@ -c $<

matrix_core_swizzle.o: matrix_core_swizzle.cpp
	$(HIPCXX) $(ICXXFLAGS) $(ICPPFLAGS) $(ALT_DEFS) -o $@ -c $<

$(EXAMPLE): permute_base.o
	$(HIPCXX) $(OFFLOAD_ARCH_FLAGS) $(ILDFLAGS) -o $@ $< $(ILDLIBS)

$(EXAMPLE_ALT): matrix_core_swizzle.o permute_alt.o
	$(HIPCXX) $(OFFLOAD_ARCH_FLAGS) $(ILDFLAGS) -o $@ $^ $(ILDLIBS)

.PHONY: clean
clean:
	$(RM) $(EXAMPLE) $(EXAMPLE_ALT) permute_base.o permute_alt.o matrix_core_swizzle.o
