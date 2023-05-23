# MIT License
#
# Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

EXAMPLE := rocsolver_getf2
COMMON_INCLUDE_DIR := ../../../Common
GPU_RUNTIME := HIP

ifneq ($(GPU_RUNTIME), HIP)
	$(error GPU_RUNTIME is set to "$(GPU_RUNTIME)". GPU_RUNTIME must be HIP.)
endif

# HIP variables
ROCM_INSTALL_DIR := /opt/rocm

HIP_INCLUDE_DIR       := $(ROCM_INSTALL_DIR)/include
ROCBLAS_INCLUDE_DIR   := $(HIP_INCLUDE_DIR)
ROCSOLVER_INCLUDE_DIR := $(HIP_INCLUDE_DIR)

CXX ?= g++

# Common variables and flags
CXX_STD   := c++17
ICXXFLAGS := -std=$(CXX_STD)
ICPPFLAGS := -isystem $(ROCBLAS_INCLUDE_DIR) -isystem $(ROCSOLVER_INCLUDE_DIR) -isystem $(HIP_INCLUDE_DIR)  -I $(COMMON_INCLUDE_DIR)
ILDFLAGS  := -L $(ROCM_INSTALL_DIR)/lib
ILDLIBS   := -lrocblas -lrocsolver -lamdhip64

CXXFLAGS  ?= -Wall -Wextra
ICPPFLAGS += -D__HIP_PLATFORM_AMD__

ICXXFLAGS += $(CXXFLAGS)
ICPPFLAGS += $(CPPFLAGS)
ILDFLAGS  += $(LDFLAGS)
ILDLIBS   += $(LDLIBS)

$(EXAMPLE): main.cpp $(COMMON_INCLUDE_DIR)/cmdparser.hpp $(COMMON_INCLUDE_DIR)/example_utils.hpp $(COMMON_INCLUDE_DIR)/rocblas_utils.hpp
	$(CXX) $(ICXXFLAGS) $(ICPPFLAGS) $(ILDFLAGS) -o $@ $< $(ILDLIBS)

clean:
	$(RM) $(EXAMPLE)

.PHONY: clean
