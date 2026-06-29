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

EXAMPLE := hipdnn_fused_conv_fprop_bias_activ
COMMON_INCLUDE_DIR := ../../../Common
EXTERNAL_DIR := ../../../External
GPU_RUNTIME ?= HIP

ifneq ($(GPU_RUNTIME), HIP)
    $(error GPU_RUNTIME is set to "$(GPU_RUNTIME)". GPU_RUNTIME must be HIP.)
endif

ROCM_PATH ?= /opt/rocm

HIP_INCLUDE_DIR := $(ROCM_PATH)/include
HIPDNN_INCLUDE_DIRS := $(shell find $(ROCM_PATH)/include/hipdnn -maxdepth 1 -type d 2>/dev/null)
HIPDNN_INCLUDE_DIRS += $(lastword $(sort $(wildcard $(ROCM_PATH)/include/hipdnn/flatbuffers_sdk/hipdnn_flatbuffers_sdk/data_objects/v*)))

CXX ?= g++

CXX_STD   := c++17
ICXXFLAGS := -std=$(CXX_STD)
ICPPFLAGS := -isystem $(HIP_INCLUDE_DIR) $(addprefix -isystem ,$(HIPDNN_INCLUDE_DIRS)) -I $(COMMON_INCLUDE_DIR) -I $(EXTERNAL_DIR) -D__HIP_PLATFORM_AMD__
ILDFLAGS  := -L $(ROCM_PATH)/lib -Wl,-rpath,$(ROCM_PATH)/lib
ILDLIBS   := -lhipdnn_backend -lamdhip64 -lpthread

CXXFLAGS ?= -Wall -Wextra

ICXXFLAGS += $(CXXFLAGS)
ICPPFLAGS += $(CPPFLAGS)
ILDFLAGS  += $(LDFLAGS)
ILDLIBS   += $(LDLIBS)

$(EXAMPLE): main.cpp $(COMMON_INCLUDE_DIR)/example_utils.hpp $(COMMON_INCLUDE_DIR)/hipdnn_utils.hpp $(EXTERNAL_DIR)/CmdParser/cmdparser.hpp
	$(CXX) $(ICXXFLAGS) $(ICPPFLAGS) $(ILDFLAGS) -o $@ $< $(ILDLIBS)

test: $(EXAMPLE)
	./$(EXAMPLE) --verify-cpu $(TEST_ARGS)

clean:
	$(RM) $(EXAMPLE)

.PHONY: clean test
