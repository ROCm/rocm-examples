# MIT License
#
# Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

EXAMPLE := hip_static_device_library
COMMON_INCLUDE_DIR := ../../Common
GPU_RUNTIME ?= HIP

ifneq ($(GPU_RUNTIME), HIP)
    $(error GPU_RUNTIME is set to "$(GPU_RUNTIME)". GPU_RUNTIME must be HIP.)
endif

# HIP variables
ROCM_PATH ?= /opt/rocm
HIP_INCLUDE_DIR  := $(ROCM_PATH)/include

HIPCXX ?= $(ROCM_PATH)/bin/amdclang++
AR ?= ar

# Common variables and flags
CXX_STD   := c++17
CXXFLAGS  ?= -Wall -Wextra
ICXXFLAGS := -std=$(CXX_STD) -O3 $(CXXFLAGS)
ICPPFLAGS := -I $(COMMON_INCLUDE_DIR) -I library $(CPPFLAGS)
ILDFLAGS  := $(LDFLAGS)
ILDLIBS   := $(LDLIBS)
include $(COMMON_INCLUDE_DIR)/filter_hip_architectures.mk
ICXXFLAGS += $(OFFLOAD_ARCH_FLAGS)

EXAMPLE_LIB := hip_static_device
LIBEXAMPLE  := lib$(EXAMPLE_LIB).a

# -x hip must come right before the HIP source, never before $(LIBEXAMPLE):
# clang would otherwise try to compile the static archive as HIP source.
$(EXAMPLE): main.hip $(LIBEXAMPLE)
	$(HIPCXX) $(ICXXFLAGS) $(ICPPFLAGS) $(ILDFLAGS) -fgpu-rdc --hip-link -o $@ $(LIBEXAMPLE) -x hip $< $(LDLIBS)

$(LIBEXAMPLE): library.o
	$(AR) rcsD $@ $^

library.o: library/library.hip
	$(HIPCXX) $(ICXXFLAGS) $(ICPPFLAGS) -c -o $@ -x hip $< -fgpu-rdc

test:

clean:
	$(RM) $(EXAMPLE) $(LIBEXAMPLE) library.o

.PHONY: all clean test
