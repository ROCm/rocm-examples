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

EXAMPLE_FWD := ComposableKernel_ck_tile_fmha_fwd
EXAMPLE_BWD := ComposableKernel_ck_tile_fmha_bwd
EXAMPLE_GENERATOR := generate.py
GPU_RUNTIME := HIP
COMMON_INCLUDE_DIR := ../../../../Common

# Supported GPU architectures for ComposableKernel FMHA
SUPPORTED_HIP_ARCHITECTURES := gfx90a gfx942
EXAMPLE := $(EXAMPLE_FWD)
include $(COMMON_INCLUDE_DIR)/filter_hip_architectures.mk

# HIP variables
ROCM_INSTALL_DIR := /opt/rocm
HIP_INCLUDE_DIR  := $(ROCM_INSTALL_DIR)/include

HIPCXX ?= $(ROCM_INSTALL_DIR)/bin/hipcc

# Common variables and flags
CXX_STD   := c++17
ICXXFLAGS := -std=$(CXX_STD) -x hip -fgpu-flush-denormals-to-zero
ICPPFLAGS :=
ILDFLAGS  :=
ILDLIBS   :=

CXXFLAGS ?= -Wall -Wextra -Wno-float-equal -Wno-undefined-func-template
ICXXFLAGS += $(CXXFLAGS)
ICPPFLAGS += $(CPPFLAGS)
ILDFLAGS  += $(LDFLAGS)
ILDLIBS   += $(LDLIBS)

FWD_PP_DEFS := -DCK_TILE_FMHA_FWD_FAST_EXP2=1
FWD_SRC_DIR := fwd_sources
FWD_OBJ_DIR := fwd_objects
BWD_SRC_DIR := bwd_sources
BWD_OBJ_DIR := bwd_objects
SRC_DIR := .

FWD_GENERATION_DONE := $(FWD_SRC_DIR)/.generated
BWD_GENERATION_DONE := $(BWD_SRC_DIR)/.generated

.PHONY: all
ifeq ($(SKIP_BUILD),1)
all:
	@echo "Skipping FMHA examples: no supported architectures"
else
all: $(EXAMPLE_FWD) $(EXAMPLE_BWD)
endif

$(FWD_SRC_DIR) $(FWD_OBJ_DIR) $(BWD_SRC_DIR) $(BWD_OBJ_DIR):
	mkdir -p $@

.PHONY: generate
generate: $(FWD_GENERATION_DONE) $(BWD_GENERATION_DONE)

$(FWD_GENERATION_DONE): $(EXAMPLE_GENERATOR) | $(FWD_SRC_DIR)
	python3 $(EXAMPLE_GENERATOR) --api fwd --receipt 2 --optdim 64 --filter "*fp16_group*" --output_dir $(FWD_SRC_DIR)
	@touch $@

$(BWD_GENERATION_DONE): $(EXAMPLE_GENERATOR) | $(BWD_SRC_DIR)
	python3 $(EXAMPLE_GENERATOR) --api bwd --receipt 3 --optdim 64 --filter "*fp16_group*" --output_dir $(BWD_SRC_DIR)
	@touch $@

FWD_SRCS := fmha_fwd.cpp
FWD_GEN_SRCS = $(shell if [ -f $(FWD_GENERATION_DONE) ]; then find $(FWD_SRC_DIR) -name "*.cpp" 2> /dev/null; fi)

BWD_SRCS := fmha_bwd.cpp
BWD_GEN_SRCS = $(shell if [ -f $(BWD_GENERATION_DONE) ]; then find $(BWD_SRC_DIR) -name "*.cpp" 2> /dev/null; fi)

-include $(FWD_GENERATION_DONE)
-include $(BWD_GENERATION_DONE)

FWD_OBJECTS := $(patsubst $(SRC_DIR)/%.cpp,$(FWD_OBJ_DIR)/%.o,$(FWD_SRCS)) \
				$(patsubst $(FWD_SRC_DIR)/%.cpp,$(FWD_OBJ_DIR)/%.o,$(FWD_GEN_SRCS))

BWD_OBJECTS := $(patsubst $(SRC_DIR)/%.cpp,$(BWD_OBJ_DIR)/%.o,$(BWD_SRCS)) \
				$(patsubst $(BWD_SRC_DIR)/%.cpp,$(BWD_OBJ_DIR)/%.o,$(BWD_GEN_SRCS))

$(FWD_OBJ_DIR)/fmha_fwd.o: fmha_fwd.cpp | $(FWD_OBJ_DIR)
	$(HIPCXX) $(ICXXFLAGS) $(ICPPFLAGS) $(FWD_PP_DEFS) -c $< -o $@

$(FWD_OBJ_DIR)/%.o: $(FWD_SRC_DIR)/%.cpp | $(FWD_OBJ_DIR)
	$(HIPCXX) -I $(shell pwd) $(ICXXFLAGS) $(ICPPFLAGS) $(FWD_PP_DEFS) -c $< -o $@

$(EXAMPLE_FWD): $(FWD_OBJECTS)
	$(HIPCXX) $(OFFLOAD_ARCH_FLAGS) $(ILDFLAGS) -o $@ $^ $(ILDLIBS)

$(BWD_OBJ_DIR)/fmha_bwd.o: fmha_bwd.cpp | $(BWD_OBJ_DIR)
	$(HIPCXX) $(ICXXFLAGS) $(ICPPFLAGS) -c $< -o $@

$(BWD_OBJ_DIR)/%.o: $(BWD_SRC_DIR)/%.cpp | $(BWD_OBJ_DIR)
	$(HIPCXX) -I $(shell pwd) $(ICXXFLAGS) $(ICPPFLAGS) -c $< -o $@

$(EXAMPLE_BWD): $(BWD_OBJECTS)
	$(HIPCXX) $(OFFLOAD_ARCH_FLAGS) $(ILDFLAGS) -o $@ $^ $(ILDLIBS)

.PHONY: clean
clean:
	$(RM) -r $(EXAMPLE_FWD) $(EXAMPLE_BWD) $(FWD_SRC_DIR) $(FWD_OBJ_DIR) $(BWD_SRC_DIR) $(BWD_OBJ_DIR)
