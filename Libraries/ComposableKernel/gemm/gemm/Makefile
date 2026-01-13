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

EXAMPLE := ComposableKernel_ck_tile_gemm
EXAMPLE_BASIC := $(EXAMPLE)_basic
EXAMPLE_UNIVERSAL := $(EXAMPLE)_universal
EXAMPLE_WEIGHT_PRESHUFFLE := $(EXAMPLE)_weight_preshuffle
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
ICXXFLAGS := -std=$(CXX_STD) -x hip -mllvm -enable-noalias-to-md-conversion=0
ICPPFLAGS :=
ILDFLAGS  :=
ILDLIBS   :=

CXXFLAGS ?= -Wall -Wextra
ICXXFLAGS += $(CXXFLAGS)
ICPPFLAGS += $(CPPFLAGS)
ILDFLAGS  += $(LDFLAGS)
ILDLIBS   += $(LDLIBS)

SOURCES := $(wildcard *.cpp)
OBJECTS := $(SOURCES:.cpp=.o)
WEIGHT_PRESHUFFLE_FLAGS := -mllvm -greedy-reverse-local-assignment=1 -mllvm --slp-threshold=-32

.PHONY: all
ifeq ($(SKIP_BUILD),1)
all:
	@echo "Skipping $(EXAMPLE): no supported architectures"
else
all: $(EXAMPLE_BASIC) $(EXAMPLE_UNIVERSAL) $(EXAMPLE_WEIGHT_PRESHUFFLE)
endif

$(EXAMPLE_BASIC): gemm_basic.cpp
	$(HIPCXX) $(ICXXFLAGS) $(ICPPFLAGS) $(OFFLOAD_ARCH_FLAGS) $(ILDFLAGS) -o $@ $< $(ILDLIBS)

$(EXAMPLE_UNIVERSAL): universal_gemm.cpp
	$(HIPCXX) $(ICXXFLAGS) $(ICPPFLAGS) $(OFFLOAD_ARCH_FLAGS) $(ILDFLAGS) -o $@ $^ $(ILDLIBS)

$(EXAMPLE_WEIGHT_PRESHUFFLE): gemm_weight_preshuffle.cpp
	$(HIPCXX) $(ICXXFLAGS) $(WEIGHT_PRESHUFFLE_FLAGS) $(ICPPFLAGS) $(OFFLOAD_ARCH_FLAGS) $(ILDFLAGS) -o $@ $^ $(ILDLIBS)

.PHONY: clean
clean:
	$(RM) $(EXAMPLE_BASIC) $(EXAMPLE_UNIVERSAL) $(EXAMPLE_WEIGHT_PRESHUFFLE) $(OBJECTS)
