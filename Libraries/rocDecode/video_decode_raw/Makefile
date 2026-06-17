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

EXAMPLE := rocdecode_video_decode_raw
COMMON_INCLUDE_DIR := ../../../Common
EXTERNAL_DIR := ../../../External
GPU_RUNTIME := HIP

include $(COMMON_INCLUDE_DIR)/find_filesystem.mk

# HIP variables
ROCM_PATH ?= /opt/rocm
UTILS_DIR := ../utils

HIP_INCLUDE_DIR       := $(ROCM_PATH)/include
ROCDECODE_INCLUDE_DIR := $(HIP_INCLUDE_DIR)

HIPCXX ?= $(ROCM_PATH)/bin/hipcc

# Common variables and flags
CXX_STD   := c++17
ICXXFLAGS := -std=$(CXX_STD)
ICPPFLAGS := -isystem $(ROCDECODE_INCLUDE_DIR) -I $(COMMON_INCLUDE_DIR) -I $(EXTERNAL_DIR) -I $(UTILS_DIR) -I $(UTILS_DIR)/rocvideodecode
ILDFLAGS  := -L $(ROCM_PATH)/lib
ILDLIBS   := -lrocdecode -lpthread $(CXX_FS_LIB)
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

SOURCES := main.cpp $(UTILS_DIR)/rocvideodecode/roc_video_dec.cpp

$(EXAMPLE): $(SOURCES) $(COMMON_INCLUDE_DIR)/example_utils.hpp $(COMMON_INCLUDE_DIR)/rocdecode_utils.hpp
	$(COMPILER) $(ICXXFLAGS) $(ICPPFLAGS) $(ILDFLAGS) -o $@ $(SOURCES) $(ILDLIBS)

# Mirrors ctest's per-codec test entries (HEVC, AVC, AV1, VP9 raw bitstreams).
# Each variant is skipped cleanly if its test video isn't present.
test: $(EXAMPLE)
	@rc=0; \
	for v in $(ROCM_PATH)/share/rocdecode/video/AMD_driving_virtual_20-H265.265 \
	         $(ROCM_PATH)/share/rocdecode/video/AMD_driving_virtual_20-H264.264 \
	         $(ROCM_PATH)/share/rocdecode/video/AMD_driving_virtual_20-AV1.ivf \
	         $(ROCM_PATH)/share/rocdecode/video/AMD_driving_virtual_20-VP9.ivf; do \
	    if [ -e "$$v" ]; then \
	        echo "==> ./$(EXAMPLE) -i $$v"; \
	        ./$(EXAMPLE) -i "$$v" || rc=$$?; \
	    else \
	        echo "==> Skipping (not found): $$v"; \
	    fi; \
	done; exit $$rc

clean:
	$(RM) $(EXAMPLE)

.PHONY: clean test
