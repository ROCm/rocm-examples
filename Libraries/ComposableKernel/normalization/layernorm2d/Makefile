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

EXAMPLE := ComposableKernel_ck_tile_layernorm2d
EXAMPLE_BASIC := ComposableKernel_ck_tile_rmsnorm2d_basic
EXAMPLE_GENERATOR := generate.py
GPU_RUNTIME := HIP
COMMON_INCLUDE_DIR := ../../../../Common

# Supported GPU architectures for ComposableKernel
SUPPORTED_HIP_ARCHITECTURES := gfx90a gfx942
include $(COMMON_INCLUDE_DIR)/filter_hip_architectures.mk

# HIP variables
ROCM_INSTALL_DIR := /opt/rocm
HIP_INCLUDE_DIR  := $(ROCM_INSTALL_DIR)/include

HIPCXX ?= $(ROCM_INSTALL_DIR)/bin/hipcc

# Common variables and flags
CXX_STD   := c++17
ICXXFLAGS := -std=$(CXX_STD) -x hip
ICPPFLAGS :=
ILDFLAGS  :=
ILDLIBS   :=

CXXFLAGS ?= -Wall -Wextra -Wno-float-equal -Wno-undefined-func-template --offload-compress
ICXXFLAGS += $(CXXFLAGS)
ICPPFLAGS += $(CPPFLAGS)
ILDFLAGS  += $(LDFLAGS)
ILDLIBS   += $(LDLIBS)

SRCS := layernorm2d_fwd.cpp
SRC_DIR := .
GEN_SRC_DIR := sources
GEN_OBJ_DIR := objects

GENERATION_DONE := $(GEN_SRC_DIR)/.generated

.PHONY: all
ifeq ($(SKIP_BUILD),1)
all:
	@echo "Skipping $(EXAMPLE): no supported architectures"
else
all: $(EXAMPLE)
endif

$(GEN_SRC_DIR) $(GEN_OBJ_DIR):
	mkdir -p $@

.PHONY: generate
generate: $(GENERATION_DONE)

$(GENERATION_DONE): $(EXAMPLE_GENERATOR) | $(GEN_SRC_DIR)
	python3 $(EXAMPLE_GENERATOR) --api fwd --working_path $(GEN_SRC_DIR) --gen_blobs
	@touch $@

GEN_SRCS = $(shell if [ -f $(GENERATION_DONE) ]; then find $(GEN_SRC_DIR) -name "*.cpp" 2> /dev/null; fi)

-include $(GENERATION_DONE)

GEN_OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(GEN_OBJ_DIR)/%.o,$(SRCS)) \
			$(patsubst $(GEN_SRC_DIR)/%.cpp,$(GEN_OBJ_DIR)/%.o,$(GEN_SRCS))
	
$(GEN_OBJ_DIR)/layernorm2d_fwd.o: layernorm2d_fwd.cpp | $(GEN_OBJ_DIR)
	$(HIPCXX) $(ICXXFLAGS) $(ICPPFLAGS) -c $< -o $@

$(GEN_OBJ_DIR)/%.o: $(GEN_SRC_DIR)/%.cpp | $(GEN_OBJ_DIR)
	$(HIPCXX) -I $(shell pwd) $(ICXXFLAGS) $(ICPPFLAGS) -c $< -o $@

$(EXAMPLE): $(GEN_OBJS)
	$(HIPCXX) $(OFFLOAD_ARCH_FLAGS) $(ILDFLAGS) -o $@ $^ $(ILDLIBS)

.PHONY: clean
clean:
	$(RM) -r $(EXAMPLE) $(GEN_SRC_DIR) $(GEN_OBJ_DIR) $(GEN_OBJS)
