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

EXAMPLE := hiptensor_contraction_bilinear_f64_f64
COMMON_INCLUDE_DIR := ../../../../../Common
EXTERNAL_DIR := ../../../../../External
GPU_RUNTIME ?= HIP

# Supported GPU architectures for hipTensor
SUPPORTED_HIP_ARCHITECTURES := gfx908 gfx90a gfx942 gfx950
include $(COMMON_INCLUDE_DIR)/filter_hip_architectures.mk

# HIP variables
ROCM_PATH ?= /opt/rocm

HIP_INCLUDE_DIR       := $(ROCM_PATH)/include
HIPBLASLT_INCLUDE_DIR := $(HIP_INCLUDE_DIR)

HIPCXX ?= $(ROCM_PATH)/bin/hipcc

# Common variables and flags
CXX_STD   := c++17
ICXXFLAGS := -std=$(CXX_STD)
ICPPFLAGS := -isystem $(HIPBLASLT_INCLUDE_DIR) -I $(COMMON_INCLUDE_DIR) -I $(EXTERNAL_DIR)
ILDFLAGS  := -L $(ROCM_PATH)/lib
ILDLIBS   := -lhiptensor

ifeq ($(GPU_RUNTIME), HIP)
  CXXFLAGS ?= -Wall -Wextra
  CPPFLAGS += -D__HIP_PLATFORM_AMD__
  COMPILER := $(HIPCXX)
else
  $(error GPU_RUNTIME is set to "$(GPU_RUNTIME)". GPU_RUNTIME must be HIP; CUDA not supported)
endif

ICXXFLAGS += $(CXXFLAGS)
ICPPFLAGS += $(CPPFLAGS)
ILDFLAGS  += $(LDFLAGS)
ILDLIBS   += $(LDLIBS)

ifeq ($(SKIP_BUILD),1)
$(EXAMPLE):
	@echo "Skipping $(EXAMPLE): no supported architectures"
else
$(EXAMPLE): main.cpp $(COMMON_INCLUDE_DIR)/example_utils.hpp $(COMMON_INCLUDE_DIR)/hiptensor_utils.hpp
	$(COMPILER) $(ICXXFLAGS) $(ICPPFLAGS) $(OFFLOAD_ARCH_FLAGS) $(ILDFLAGS) -o $@ $< $(ILDLIBS)
endif

ifeq ($(SKIP_BUILD),1)
test:
	@echo "Skipping $(EXAMPLE) test: no supported architectures"
else
test: $(EXAMPLE)
	./$(EXAMPLE) $(TEST_ARGS)
endif

clean:
	$(RM) $(EXAMPLE)

.PHONY: clean test
