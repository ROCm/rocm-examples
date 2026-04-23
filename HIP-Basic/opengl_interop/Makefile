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

EXAMPLE := hip_opengl_interop
COMMON_INCLUDE_DIR := ../../Common
EXTERNAL_DIR := ../../External
GPU_RUNTIME ?= HIP

# Dependency checking
REQUIRED_DEPS := GLFW3
include $(COMMON_INCLUDE_DIR)/require_deps.mk

ifneq ($(SKIP_BUILD),1)

# HIP variables
ROCM_PATH ?= /opt/rocm
HIP_INCLUDE_DIR  := $(ROCM_PATH)/include

HIPCXX ?= $(ROCM_PATH)/bin/hipcc

# Common variables and flags
CXX_STD   := c++17
ICXXFLAGS := -std=$(CXX_STD)
ICPPFLAGS := -I $(COMMON_INCLUDE_DIR) -I $(EXTERNAL_DIR) $(GLFW3_CFLAGS)
ILDFLAGS  :=
ILDLIBS   :=  $(GLFW3_LIBS) -ldl

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

SRCS := main.hip $(EXTERNAL_DIR)/glad/glad.cpp

$(EXAMPLE): $(SRCS) $(COMMON_INCLUDE_DIR)/example_utils.hpp
	$(HIPCXX) $(ICXXFLAGS) $(ICPPFLAGS) $(ILDFLAGS) -o $@ $(SRCS) $(ILDLIBS)

$(EXTERNAL_DIR)/glad/glad.h: $(EXTERNAL_DIR)/KHR/khrplatform.h

$(EXTERNAL_DIR)/glad/glad.cpp: $(EXTERNAL_DIR)/glad/glad.h

test:

clean:
	$(RM) $(EXAMPLE)

.PHONY: clean test

else
all:
clean:
test:
.PHONY: all clean test
endif
