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

EXAMPLE := rocdecode_video_decode
COMMON_INCLUDE_DIR := ../../../Common
EXTERNAL_DIR := ../../../External
GPU_RUNTIME := HIP

# HIP variables
ROCM_INSTALL_DIR ?= /opt/rocm
UTILS_DIR := ${ROCM_INSTALL_DIR}/share/rocdecode/utils

HIP_INCLUDE_DIR       := $(ROCM_INSTALL_DIR)/include
ROCDECODE_INCLUDE_DIR := $(HIP_INCLUDE_DIR)

HIPCXX ?= $(ROCM_INSTALL_DIR)/bin/hipcc

# Detect which rocdecode host library is available
ROCDECODE_HOST_LIB := $(shell if [ -f $(ROCM_INSTALL_DIR)/lib/librocdecode-host.so ]; then \
                                  echo "rocdecode-host"; \
                              elif [ -f $(ROCM_INSTALL_DIR)/lib/librocdecodehost.so ]; then \
                                  echo "rocdecodehost"; \
                              fi)

# Common variables and flags
CXX_STD   := c++17
ICXXFLAGS := -std=$(CXX_STD)
ICPPFLAGS := -isystem $(ROCDECODE_INCLUDE_DIR) -I $(COMMON_INCLUDE_DIR) -I $(EXTERNAL_DIR) -I .. -I $(UTILS_DIR) -I $(UTILS_DIR)/rocvideodecode
ILDFLAGS  := -L $(ROCM_INSTALL_DIR)/lib
ILDLIBS   := -lrocdecode -lrocprofiler-register $(shell pkg-config --libs libavcodec libavformat libavutil 2>/dev/null || echo "-lavcodec -lavformat -lavutil") -lpthread

ifeq ($(GPU_RUNTIME), HIP)
	CXXFLAGS ?= -Wall -Wextra
	CPPFLAGS += -D__HIP_PLATFORM_AMD__

	# Add host decode library and flag if available
	ifneq ($(ROCDECODE_HOST_LIB),)
		ILDLIBS  += -l$(ROCDECODE_HOST_LIB)
		CPPFLAGS += -DENABLE_HOST_DECODE=1
		ICPPFLAGS += -I $(UTILS_DIR)/ffmpegvideodecode
	else
		CPPFLAGS += -DENABLE_HOST_DECODE=0
	endif

	# FFmpeg version detection
	AVCODEC_VERSION := $(shell pkg-config --modversion libavcodec 2>/dev/null || echo "0")
	AVCODEC_MAJOR := $(shell echo $(AVCODEC_VERSION) | cut -d. -f1)
	AVCODEC_MINOR := $(shell echo $(AVCODEC_VERSION) | cut -d. -f2)
	AVCODEC_PATCH := $(shell echo $(AVCODEC_VERSION) | cut -d. -f3)

	# Compare version (58.134.100)
	ifeq ($(shell test $(AVCODEC_MAJOR) -lt 58 || \
	              (test $(AVCODEC_MAJOR) -eq 58 && test $(AVCODEC_MINOR) -lt 134) || \
	              (test $(AVCODEC_MAJOR) -eq 58 && test $(AVCODEC_MINOR) -eq 134 && test $(AVCODEC_PATCH) -le 100); echo $$?),0)
		CPPFLAGS += -DUSE_AVCODEC_GREATER_THAN_58_134=0
	else
		CPPFLAGS += -DUSE_AVCODEC_GREATER_THAN_58_134=1
	endif

	COMPILER := $(HIPCXX)
else
	$(error GPU_RUNTIME is set to "$(GPU_RUNTIME)". GPU_RUNTIME must be HIP; CUDA not supported)
endif

ICXXFLAGS += $(CXXFLAGS)
ICPPFLAGS += $(CPPFLAGS)
ILDFLAGS  += $(LDFLAGS)
ILDLIBS   += $(LDLIBS)

SOURCES := main.cpp $(UTILS_DIR)/rocvideodecode/roc_video_dec.cpp

ifneq ($(ROCDECODE_HOST_LIB),)
	SOURCES += $(UTILS_DIR)/ffmpegvideodecode/ffmpeg_video_dec.cpp
endif

$(EXAMPLE): $(SOURCES) $(COMMON_INCLUDE_DIR)/example_utils.hpp $(COMMON_INCLUDE_DIR)/rocdecode_utils.hpp
	$(COMPILER) $(ICXXFLAGS) $(ICPPFLAGS) $(ILDFLAGS) -o $@ $(SOURCES) $(ILDLIBS)

clean:
	$(RM) $(EXAMPLE)

.PHONY: clean
