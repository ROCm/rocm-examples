# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation of rights
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

EXAMPLE := roccv_normalize
COMMON_INCLUDE_DIR := ../../../Common
EXTERNAL_DIR := ../../../External
GPU_RUNTIME := HIP

# Dependency checking
REQUIRED_DEPS := OPENCV
include $(COMMON_INCLUDE_DIR)/require_deps.mk

ifneq ($(SKIP_BUILD),1)

# HIP variables
ROCM_PATH := /opt/rocm

HIP_INCLUDE_DIR := $(ROCM_PATH)/include
ROCCV_INCLUDE_DIR := $(HIP_INCLUDE_DIR)/roccv

HIPCXX ?= $(ROCM_PATH)/bin/hipcc

# OpenCV variables
OPENCV_CXXFLAGS := $(OPENCV_CFLAGS)
OPENCV_LDFLAGS := $(OPENCV_LIBS)

# Common variables and flags
CXX_STD   := c++20
ICXXFLAGS := -std=$(CXX_STD)
ICPPFLAGS := -isystem $(ROCCV_INCLUDE_DIR) -I $(COMMON_INCLUDE_DIR) -I $(EXTERNAL_DIR) -DEXAMPLE_DATA=\"$(CURDIR)/../data/mug_422.jpg\"
ILDFLAGS  := -L $(ROCM_PATH)/lib
ILDLIBS   := -lroccv

ifeq ($(GPU_RUNTIME), HIP)
	CXXFLAGS ?= -Wall -Wextra
	CPPFLAGS += -D__HIP_PLATFORM_AMD__
	COMPILER := $(HIPCXX)
else
	$(error GPU_RUNTIME is set to "$(GPU_RUNTIME)". GPU_RUNTIME must be HIP; CUDA not supported)
endif

ICXXFLAGS += $(CXXFLAGS) $(OPENCV_CXXFLAGS)
ICPPFLAGS += $(CPPFLAGS)
ILDFLAGS  += $(LDFLAGS) $(OPENCV_LDFLAGS)
ILDLIBS   += $(LDLIBS)

$(EXAMPLE): main.cpp $(COMMON_INCLUDE_DIR)/example_utils.hpp
	$(COMPILER) $(ICXXFLAGS) $(ICPPFLAGS) $(ILDFLAGS) -o $@ $< $(ILDLIBS)

clean:
	$(RM) $(EXAMPLE)

.PHONY: clean

else
all:
clean:
.PHONY: all clean
endif
