# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

EXAMPLE     := hipfile_roundtrip_async_multi_stream_registered
COMMON_DIR  := ../../common
GPU_RUNTIME ?= HIP

ROCM_PATH ?= /opt/rocm
HIP_INCLUDE_DIR := $(ROCM_PATH)/include

HIPCXX ?= $(ROCM_PATH)/bin/hipcc

CXX_STD   := c++17
ICXXFLAGS := -std=$(CXX_STD)
ICPPFLAGS := -I $(COMMON_DIR)
ILDFLAGS  :=
ILDLIBS   := -lhipfile

ifeq ($(GPU_RUNTIME), CUDA)
  ICXXFLAGS += -x cu
  ICPPFLAGS += -isystem $(HIP_INCLUDE_DIR)
else ifeq ($(GPU_RUNTIME), HIP)
  CXXFLAGS ?= -Wall -Wextra
else
  $(error GPU_RUNTIME is set to "$(GPU_RUNTIME)". GPU_RUNTIME must be either CUDA or HIP)
endif

ICXXFLAGS += $(CXXFLAGS)
ICPPFLAGS += $(CPPFLAGS)
ILDFLAGS  += $(LDFLAGS)
ILDLIBS   += $(LDLIBS)

SRCS := roundtrip-async-multi-stream-registered.cpp $(COMMON_DIR)/examples_common.cpp

$(EXAMPLE): $(SRCS)
	$(HIPCXX) $(ICXXFLAGS) $(ICPPFLAGS) $(ILDFLAGS) -o $@ $(SRCS) $(ILDLIBS)

test: $(EXAMPLE)
	./$(EXAMPLE) roundtrip_async_msr_read.bin roundtrip_async_msr_write.bin $(TEST_ARGS)

clean:
	$(RM) $(EXAMPLE) roundtrip_async_msr_read.bin roundtrip_async_msr_write.bin

.PHONY: clean test
