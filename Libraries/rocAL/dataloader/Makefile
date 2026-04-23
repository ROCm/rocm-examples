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

EXAMPLE := rocal_dataloader
EXTERNAL_DIR := ../../../External
GPU_RUNTIME := HIP

# HIP variables
ROCM_PATH ?= /opt/rocm

HIP_INCLUDE_DIR  := $(ROCM_PATH)/include
ROCAL_INCLUDE_DIR := $(HIP_INCLUDE_DIR)/rocal
OPENCV_INCLUDE_DIR := /usr/include/opencv4

CXX ?= g++

# Common variables and flags
CXX_STD   := c++17
ICXXFLAGS := -std=$(CXX_STD)
ICPPFLAGS := -isystem $(HIP_INCLUDE_DIR) -isystem $(ROCAL_INCLUDE_DIR) -isystem $(OPENCV_INCLUDE_DIR) -I $(EXTERNAL_DIR)
ILDFLAGS  := -L $(ROCM_PATH)/lib
ILDLIBS   := -lrocal -lturbojpeg -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -lpthread

ifeq ($(GPU_RUNTIME), HIP)
	CXXFLAGS ?= -Wall -Wextra
	CPPFLAGS += -D__HIP_PLATFORM_AMD__
	COMPILER := $(CXX)
else
	$(error GPU_RUNTIME is set to "$(GPU_RUNTIME)". GPU_RUNTIME must be HIP)
endif

ICXXFLAGS += $(CXXFLAGS)
ICPPFLAGS += $(CPPFLAGS) -DEXAMPLE_DATA_DIR=\"$(CURDIR)/../data\"
ILDFLAGS  += $(LDFLAGS)
ILDLIBS   += $(LDLIBS)

$(EXAMPLE): main.cpp
	$(COMPILER) $(ICXXFLAGS) $(ICPPFLAGS) $(ILDFLAGS) -o $@ $< $(ILDLIBS)

test: $(EXAMPLE)
	./$(EXAMPLE) $(TEST_ARGS)

clean:
	$(RM) $(EXAMPLE)

.PHONY: clean test
