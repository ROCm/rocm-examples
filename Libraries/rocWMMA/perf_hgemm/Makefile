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

EXAMPLE := rocwmma_perf_hgemm
COMMON_INCLUDE_DIR := ../../../Common
GPU_RUNTIME ?= HIP

ifneq ($(GPU_RUNTIME), HIP)
  $(error GPU_RUNTIME is set to "$(GPU_RUNTIME)". GPU_RUNTIME must be HIP.)
endif

# HIP variables
ROCM_INSTALL_DIR ?= /opt/rocm

HIP_INCLUDE_DIR       := $(ROCM_INSTALL_DIR)/include
ROCWMMA_INCLUDE_DIR := $(HIP_INCLUDE_DIR)

HIPCXX ?= $(ROCM_INSTALL_DIR)/bin/hipcc

# Common variables and flags
CXX_STD   := c++17
CXXFLAGS  ?= -Wall -Wextra
ICXXFLAGS := -std=$(CXX_STD) $(CXXFLAGS)
ICPPFLAGS := -I $(COMMON_INCLUDE_DIR) -isystem $(ROCWMMA_INCLUDE_DIR) -D__HIP_PLATFORM_AMD__ $(CPPFLAGS)
ILDFLAGS  := $(LDFLAGS)
ILDLIBS   := $(LDLIBS)

$(EXAMPLE): main.hip $(COMMON_INCLUDE_DIR)/example_utils.hpp
	$(HIPCXX) $(ICXXFLAGS) $(ICPPFLAGS) $(ILDFLAGS) -o $@ $< $(ILDLIBS)

clean:
	$(RM) $(EXAMPLE)

.PHONY: clean
